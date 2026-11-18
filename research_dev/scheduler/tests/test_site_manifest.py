"""The site manifest: strict round trip, validation, and what discovery may and may not change."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler.adapters.contracts import PhysicalAdapterError  # noqa: E402
from research_dev.scheduler.configuration.common import SchedulerConfigurationError  # noqa: E402
from research_dev.scheduler.configuration.site import SiteManifest  # noqa: E402
from research_dev.scheduler.drivers import discover  # noqa: E402


def site_document() -> dict:
    return {
        "schema": "research-scheduler-site-v1",
        "site_id": "site-a",
        "host": {"host_id": "site-a", "cpu_model": "Synthetic CPU 64", "logical_cpus": 64,
                 "memory_total_bytes": 128 << 30,
                 "gpus": [{"index": 0, "uuid": "GPU-0000", "name": "Synthetic GPU", "memory_total_bytes": 48 << 30},
                          {"index": 1, "uuid": "GPU-1111", "name": "Synthetic GPU", "memory_total_bytes": 48 << 30}],
                 "rapl_zones": ["intel-rapl:0"], "serving_gpu_index": 1, "threads": 32},
        "phones": [
            {"device_id": "phone-a", "serial": "SERIALA", "adb_port": 5037, "model": "Model A", "soc": "SM8850",
             "kernel_release": "6.12-a", "memory_total_bytes": 16 << 30, "root": True, "usb_sysfs_device": "8-3",
             "usb_speed_mbps": 5000, "transport": "functionfs-bridge", "kind": "htp", "backend": "HTP0",
             "bundle_directory": "/data/local/tmp/bundle-a", "work_directory": "/data/local/tmp/site-a",
             "worker_port": 27311, "forward_port": 27311, "column_quantum": 3200, "max_tokens": 4,
             "worker_environment": {"GGML_HEXAGON_NHVX": "4"}, "sessions": 3, "session_limit_bytes": 3300000000,
             "as_root": True, "minimum_usb_speed_mbps": 5000,
             "power": {"usb_current_node": "usb/current_now", "usb_voltage_node": "usb/voltage_now",
                       "battery_current_node": "battery/current_now", "battery_voltage_node": "battery/voltage_now",
                       "battery_current_unit": "mA", "battery_discharge_sign": 1, "gauge_ratio": 0.42,
                       "gauge_ratio_provenance": "calibrated", "needs_root": True},
             "functionfs": {"kernel_release": "6.12-a", "bridge_port": 27391, "session_script": "/data/x.sh"}},
            {"device_id": "phone-b", "serial": "SERIALB", "adb_port": 5037, "model": "Model B", "soc": "SM8650",
             "kernel_release": "6.1-b", "memory_total_bytes": 12 << 30, "root": False, "usb_sysfs_device": "5-4",
             "usb_speed_mbps": 480, "transport": "adb-tcp", "kind": "htp", "backend": "HTP0",
             "bundle_directory": "/data/local/tmp/bundle-b", "work_directory": "/data/local/tmp/site-a",
             "worker_port": 27321, "forward_port": 27321, "column_quantum": 3200, "max_tokens": 4,
             "power": {"assumed_active_w": 4.5, "assumed_idle_w": 0.875}},
        ],
        "paths": {"repo_root": "/repo", "server_path": "/repo/bin/llama-server", "adb_path": "/opt/adb",
                  "work_root": "/work/site-a", "models": {"model-a": "/models/model-a.gguf"},
                  "host_tools": {"usb_bridge": "/repo/bin/usb-bridge"}},
    }


def load(document: dict) -> SiteManifest:
    return SiteManifest.from_json(document, base=Path("/base"))


class SiteManifestTests(unittest.TestCase):
    def test_round_trip_is_stable_and_omits_defaults(self):
        site = load(site_document())
        again = load(site.to_json())
        self.assertEqual(again, site)
        self.assertEqual(again.to_json(), site.to_json())
        row = site.to_json()["phones"][1]
        for name in ("sessions", "session_limit_bytes", "as_root", "functionfs", "worker_environment"):
            self.assertNotIn(name, row)
        self.assertEqual(site.host.serving_gpu.uuid, "GPU-1111")
        self.assertEqual(site.phone("phone-b").power.measured, False)
        self.assertEqual(site.phone("phone-a").functionfs["bridge_port"], 27391)
        self.assertEqual(site.paths.models["model-a"], Path("/models/model-a.gguf"))
        with self.assertRaisesRegex(SchedulerConfigurationError, "no phone"):
            site.phone("phone-c")

    def test_file_round_trip_and_relative_paths(self):
        document = site_document()
        document["paths"]["work_root"] = "work"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "site.json"
            path.write_text(json.dumps(document), encoding="ascii")
            site = SiteManifest.load(path)
            self.assertEqual(site.paths.work_root, (Path(directory) / "work").resolve())
            with self.assertRaisesRegex(SchedulerConfigurationError, "cannot be loaded"):
                SiteManifest.load(Path(directory) / "absent.json")

    def test_unknown_keys_are_refused_everywhere(self):
        for edit in (lambda d: d.update(extra=1), lambda d: d["host"].update(extra=1),
                     lambda d: d["host"]["gpus"][0].update(extra=1), lambda d: d["phones"][0].update(extra=1),
                     lambda d: d["phones"][0]["power"].update(extra=1), lambda d: d["paths"].update(extra=1)):
            document = site_document()
            edit(document)
            with self.assertRaisesRegex(SchedulerConfigurationError, "unknown keys: extra"):
                load(document)

    def test_inconsistent_sites_are_refused(self):
        cases = (
            (lambda d: d["host"].update(serving_gpu_index=2), "names no GPU"),
            (lambda d: d["host"].update(threads=65), "exceed the logical CPUs"),
            (lambda d: d["host"]["gpus"][1].update(index=0), "distinct indices"),
            (lambda d: d["phones"][1].update(serial="SERIALA"), "serials are not unique"),
            (lambda d: d["phones"][1].update(forward_port=27311), "forward ports are not unique"),
            (lambda d: d["phones"][1].update(transport="carrier-pigeon"), "transport must be one of"),
            (lambda d: d["phones"][1].update(kind="quantum"), "kind must be one of"),
            (lambda d: d["phones"][1].update(as_root=True), "not rooted"),
            (lambda d: d["phones"][1].update(sessions=2), "need the direct USB transport"),
            (lambda d: d["phones"][0].pop("functionfs"), "needs root and a functionfs section"),
            (lambda d: d["phones"][0].update(bundle_directory="relative/dir"), "absolute phone path"),
            (lambda d: d["phones"][0]["worker_environment"].update(LD_LIBRARY_PATH="/x"), "LD_LIBRARY_PATH"),
            (lambda d: d["phones"][0]["power"].pop("usb_voltage_node"), "all four nodes or none"),
            (lambda d: d["phones"][1]["power"].pop("assumed_idle_w"), "both assumed powers or none"),
            (lambda d: d["phones"][1].update(power={}), "neither measured nor assumed"),
            (lambda d: d["phones"][0]["power"].update(battery_current_unit="A"), "mA or uA"),
            (lambda d: d["phones"][0]["power"].update(battery_discharge_sign=0), "sign must be"),
            (lambda d: d["phones"][1].update(power={**site_document()["phones"][0]["power"]}), "need root"),
            (lambda d: d.update(schema="other"), "schema differs"),
        )
        for edit, message in cases:
            document = site_document()
            edit(document)
            with self.subTest(message=message), self.assertRaisesRegex(SchedulerConfigurationError, message):
                load(document)


class DiscoveryMergeTests(unittest.TestCase):
    def test_refresh_replaces_facts_and_keeps_declared_fields(self):
        before = site_document()
        host = {"cpu_model": "Synthetic CPU 64", "logical_cpus": 64, "memory_total_bytes": 128 << 30,
                "gpus": copy.deepcopy(before["host"]["gpus"]), "rapl_zones": []}
        phone = {"model": "Model A", "soc": "SM8850", "kernel_release": "6.12-stock", "memory_total_bytes": 16 << 30,
                 "root": True, "usb_sysfs_device": "2-1", "usb_speed_mbps": 5000}
        after = discover.refresh(before, host, {"SERIALA": phone})
        self.assertEqual(sorted(discover.changes(before, after)), [
            "host rapl_zones: ['intel-rapl:0'] -> []",
            "phone phone-a kernel_release: '6.12-a' -> '6.12-stock'",
            "phone phone-a usb_sysfs_device: '8-3' -> '2-1'"])
        self.assertEqual(after["host"]["threads"], 32)
        self.assertEqual(after["phones"][0]["transport"], "functionfs-bridge")
        self.assertEqual(after["phones"][1], before["phones"][1])       # no facts given: untouched
        self.assertEqual(before["phones"][0]["usb_sysfs_device"], "8-3")  # the input is not modified
        load(after)
        with self.assertRaisesRegex(PhysicalAdapterError, "not a site manifest"):
            discover.refresh({"schema": "x"}, host, {})

    def test_new_site_and_added_phone_load(self):
        source = site_document()
        host = {name: source["host"][name] for name in ("cpu_model", "logical_cpus", "memory_total_bytes", "gpus",
                                                        "rapl_zones")}
        document = discover.new_site_document("site-b", host, serving_gpu_index=0, threads=16, paths=source["paths"])
        self.assertEqual(load(document).phones, ())
        row = dict(source["phones"][1])
        facts = {name: row.pop(name) for name in ("model", "soc", "kernel_release", "memory_total_bytes", "root",
                                                  "usb_sysfs_device", "usb_speed_mbps")}
        document = discover.add_phone(document, facts, row)
        self.assertEqual(load(document).phone("phone-b").usb_speed_mbps, 480)
        self.assertEqual(discover.phone_serials(document), ["SERIALB"])
        with self.assertRaisesRegex(PhysicalAdapterError, "already has this phone"):
            discover.add_phone(document, facts, row)


if __name__ == "__main__":
    unittest.main()
