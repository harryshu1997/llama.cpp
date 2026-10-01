"""WS10 opt-in wiring of ``helper_phones[].transport: "aoa-bridge"``.

Without the value every rig row, declaration, identity and evidence bundle is unchanged (byte-identical JSON,
same session class); with it the co-helper declaration carries ``ffn_link_transport: aoa-bridge`` while the
server-facing contract (and llama-server's environment) stays the adb-tcp TCP endpoint, the transport identity
pins the relay, the host bridge and the keep-awake options, and the evidence builds an AOA bridge session.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from research_dev.scheduler._internal.plan_contracts.co_helpers import RuntimeCoHelperPhone  # noqa: E402
from research_dev.scheduler._internal.plan_contracts.common import RuntimePlanError  # noqa: E402
from research_dev.scheduler.adapters.co_helper_lifecycle import CoHelperLifecycle  # noqa: E402
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError  # noqa: E402
from research_dev.scheduler.adapters.phone_aoa_session import (  # noqa: E402
    AoaBridgeConfiguration, AoaBridgePhoneWorkerSession,
)
from research_dev.scheduler.adapters.phone_helpers import (  # noqa: E402
    IDENTITY_REQUIREMENTS, PhoneHelperBinding, PhoneHelperTransportIdentity, helper_server_environment,
)
from research_dev.scheduler.adapters.phone_tcp_session import AdbTcpPhoneWorkerSession  # noqa: E402
from research_dev.scheduler.adapters.phone_transport import (  # noqa: E402
    AOA_BRIDGE_TRANSPORT_GENERATION, phone_transport_contract,
)
from research_dev.scheduler.campaigns.burstgpt import catalog as campaign_catalog  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.helper_phone_evidence import (  # noqa: E402
    digest, load_helper_evidence, validate_helper_declaration,
)
from research_dev.scheduler.configuration.common import SchedulerConfigurationError  # noqa: E402
from research_dev.scheduler.configuration.rig import RigManifest  # noqa: E402

import two_phone_harness as h  # noqa: E402
import test_static_helper_campaign as static_campaign  # noqa: E402
from test_two_phone_helpers import two_phone_rig_json  # noqa: E402

SHA = "sha256:" + "b" * 64
AOA = {"usb_sysfs_device": "2-9.2", "relay_path": "/data/local/tmp/s43-aoa/s43-aoa-relay", "relay_sha256": SHA,
       "bridge_script_sha256": SHA, "relay_options": {"qos_latency_us": 0, "qos_window_ms": 1500, "cpus": "03"},
       "bridge_options": {"keepalive_ms": 2}}


class RigAndCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = h.manifest(self.directory.name)
        self.index = Path(self.directory.name) / "FFN_SHARDS.json"
        self.index.write_text(json.dumps({"parent_sha256": self.model.artifact_sha256,
                                          "schema": "s42-ffn-shard-index-v1", "shards": [{
            "columns": 128, "layer_mask": "%016x" % h.PIXEL_MASK, "n_ff": 128,
            "parent_sha256": self.model.artifact_sha256, "path": "HTP0.ffn.gguf", "session_id": "HTP0",
            "shard_bytes": 4096, "shard_sha256": h.SHARD_SHA, "weight_type": "F16"}]}), encoding="ascii")
        self.config = SimpleNamespace(
            helper_phone_ffn_shards={"pixel10pro-phone": (self.index, "/data/local/tmp/pixel-shards")},
            phone_ffn_shard_index_path=None, phone_ffn_shard_directory=None)

    def rig_json(self, aoa: dict | None) -> dict:
        row = two_phone_rig_json()
        row["helper_phones"][0]["forward_port"] = 26991
        if aoa is not None:
            row["helper_phones"][0]["transport"] = "aoa-bridge"
            row["helper_phones"][0]["aoa_bridge"] = aoa
        return row

    def test_absent_transport_value_keeps_rig_and_declaration_byte_identical(self) -> None:
        row = self.rig_json(None)
        rig = RigManifest.from_json(row, Path("/"))
        self.assertIsNone(rig.helper_phones[0].aoa_bridge)
        self.assertNotIn("aoa_bridge", rig.helper_phones[0].to_json())
        self.assertEqual(json.dumps(rig.to_json(), sort_keys=True), json.dumps(row, sort_keys=True))
        helper, = campaign_catalog.helper_phone_co_helpers(rig, self.config, self.model).helpers
        self.assertNotIn("ffn_link_transport", helper.transport_parameters)

    def test_aoa_bridge_row_round_trips_and_marks_the_link_but_not_the_server(self) -> None:
        row = self.rig_json(AOA)
        rig = RigManifest.from_json(row, Path("/"))
        self.assertEqual(rig.helper_phones[0].transport, "aoa-bridge")
        self.assertEqual(json.dumps(rig.to_json(), sort_keys=True), json.dumps(row, sort_keys=True))
        declaration = campaign_catalog.helper_phone_co_helpers(rig, self.config, self.model)
        helper, = declaration.helpers
        self.assertEqual(dict(helper.transport_parameters), {
            "adb_port": 5037, "adb_serial": h.PIXEL_SERIAL, "ffn_transport": "adb-tcp",
            "ffn_worker_host": "127.0.0.1", "ffn_worker_port": 26991, "ffn_link_transport": "aoa-bridge",
            "phone_worker_port": 26990})
        plain = campaign_catalog.helper_phone_co_helpers(RigManifest.from_json(self.rig_json(None), Path("/")),
                                                         self.config, self.model).helpers[0]
        # llama-server's helper environment is the adb-tcp one, byte for byte
        op15 = PhoneHelperBinding(device_id="op15-phone", serial=h.OP15_SERIAL,
                                  layer_mask=1 << h.PIXEL_MASK.bit_length(), label="op15")
        environments = []
        for row_helper in (helper, plain):
            contract = phone_transport_contract(row_helper.transport_parameters)
            environments.append(dict(helper_server_environment(
                (op15, PhoneHelperBinding(device_id=row_helper.device_id, serial=row_helper.serial,
                                          layer_mask=row_helper.layer_mask, label="pixel",
                                          transport_parameters=dict(row_helper.transport_parameters))),
                {"S41_SERVER_FFN_LAYER_MASK": str(op15.layer_mask | h.PIXEL_MASK)},
                (phone_transport_contract(dict(h.FUNCTIONFS_PARAMETERS)), contract))))
        self.assertEqual(environments[0], environments[1])
        self.assertEqual(environments[0]["S41_SERVER_FFN_HELPER1_TRANSPORT"], "tcp")

    def test_rig_refuses_inconsistent_aoa_rows(self) -> None:
        for mutate, message in (
            (lambda row: row.pop("aoa_bridge"), "aoa_bridge settings"),
            (lambda row: row.update(transport="adb-tcp"), "aoa_bridge settings"),
            (lambda row: row.update(forward_port=0), "fixed forward port"),
            (lambda row: row.update(link_delay_proxy_port=26992), "no link delay proxy"),
            (lambda row: row["aoa_bridge"].update(surprise=1), "unknown keys"),
            (lambda row: row["aoa_bridge"].pop("relay_sha256"), "lacks relay_sha256"),
            (lambda row: row.update(transport="usb"), "helper phone transport"),
        ):
            row = self.rig_json(json.loads(json.dumps(AOA)))
            mutate(row["helper_phones"][0])
            with self.subTest(message=message), self.assertRaisesRegex(SchedulerConfigurationError, message):
                RigManifest.from_json(row, Path("/"))

    def test_co_helper_contract_accepts_only_the_aoa_link_value(self) -> None:
        helper = h.co_helpers().helpers[0]

        def phone(**parameters):
            return RuntimeCoHelperPhone(
                device_id=helper.device_id, serial=helper.serial, label=helper.label, session_id=helper.session_id,
                layer_mask=helper.layer_mask, column_quantum=helper.column_quantum, max_tokens=helper.max_tokens,
                shard_sha256=helper.shard_sha256, resident_bytes=helper.resident_bytes,
                transport_parameters={**h.PIXEL_TRANSPORT, **parameters})
        self.assertEqual(phone(ffn_link_transport="aoa-bridge").transport_parameters["ffn_link_transport"],
                         "aoa-bridge")
        for parameters in ({"ffn_link_transport": "wifi"},
                           {"ffn_link_transport": "aoa-bridge", "ffn_link_proxy_upstream_port": 26992}):
            with self.subTest(parameters=parameters), self.assertRaisesRegex(RuntimePlanError, "aoa-bridge"):
                phone(**parameters)

    def test_lifecycle_refuses_a_session_of_the_other_link(self) -> None:
        rig = RigManifest.from_json(self.rig_json(AOA), Path("/"))
        declaration = campaign_catalog.helper_phone_co_helpers(rig, self.config, self.model)
        helper, = declaration.helpers

        class Worker:
            def __init__(self, link: str | None) -> None:
                self.configuration = SimpleNamespace(
                    device_id=helper.device_id, serial=helper.serial, layer_mask=helper.layer_mask,
                    column_quantum=helper.column_quantum, max_tokens=helper.max_tokens, phone_port=26990,
                    forward_port=26991)
                self.link = link

            def preflight(self):
                return SimpleNamespace(to_json=lambda: {"kind": "preflight"})

            def start(self, log_path):
                return SimpleNamespace(to_json=lambda: {"kind": "start"})

            def transport_parameters(self):
                return {**h.PIXEL_TRANSPORT, "ffn_worker_port": 26991,
                        **({"ffn_link_transport": self.link} if self.link else {})}

        CoHelperLifecycle(declaration, {helper.device_id: Worker("aoa-bridge")})._start(helper, Path("/w.log"))
        with self.assertRaisesRegex(PhysicalAdapterError, "forward differs"):
            CoHelperLifecycle(declaration, {helper.device_id: Worker(None)})._start(helper, Path("/w.log"))
        plain = campaign_catalog.helper_phone_co_helpers(RigManifest.from_json(self.rig_json(None), Path("/")),
                                                         self.config, self.model)
        with self.assertRaisesRegex(PhysicalAdapterError, "forward differs"):
            CoHelperLifecycle(plain, {helper.device_id: Worker("aoa-bridge")})._start(plain.helpers[0],
                                                                                      Path("/w.log"))


class IdentityAndEvidenceTests(unittest.TestCase):
    IDENTITY = {
        "device_id": "pixel10pro-phone", "transport": "aoa-bridge",
        "transport_generation": AOA_BRIDGE_TRANSPORT_GENERATION, "minimum_usb_speed_mbps": 5000,
        "hardware_identity": {"adb_usb_identity": "18d1:4ee7", "aoa_usb_identity": "18d1:2d01",
                              "host_usb_controller": "0000:00:14.0", "phone_kernel_release": "6.6.102",
                              "phone_usb_serial": h.PIXEL_SERIAL, "phone_usb_sysfs_device": "2-9.2"},
        "software_identity": {"aoa_bridge_options_sha256": SHA, "host_binary_sha256": SHA, "host_bridge_sha256": SHA,
                              "phone_relay_sha256": SHA, "phone_shard_sha256": SHA, "phone_worker_sha256": SHA,
                              "transport_client_source_sha256": SHA, "worker_environment_sha256": SHA,
                              "phone_library_sha256:libggml.so": SHA},
        "receipts": {},
    }

    def test_aoa_identity_needs_its_own_pins_generation_and_receipts(self) -> None:
        identity = PhoneHelperTransportIdentity(**self.IDENTITY)
        self.assertEqual(identity.missing_receipts, IDENTITY_REQUIREMENTS["aoa-bridge"]["receipts"])
        complete = PhoneHelperTransportIdentity(**{**self.IDENTITY, "receipts": {
            kind: SHA for kind in IDENTITY_REQUIREMENTS["aoa-bridge"]["receipts"]}})
        self.assertTrue(complete.qualified)
        self.assertEqual(PhoneHelperTransportIdentity.from_json(complete.to_json()), complete)
        for change in ({"transport_generation": "adb-tcp-worker-v6"},
                       {"hardware_identity": {**self.IDENTITY["hardware_identity"], "aoa_usb_identity": ""}},
                       {"software_identity": {key: value for key, value in self.IDENTITY["software_identity"].items()
                                              if key != "phone_relay_sha256"}},
                       {"software_identity": {key: value for key, value in self.IDENTITY["software_identity"].items()
                                              if not key.startswith("phone_library")}}):
            with self.subTest(change=list(change)), self.assertRaises(PhysicalAdapterError):
                PhoneHelperTransportIdentity(**{**self.IDENTITY, **change})
        # the adb-tcp message is unchanged by the generation table
        with self.assertRaisesRegex(PhysicalAdapterError, "^adb-tcp identity generation is adb-tcp-worker-v6$"):
            PhoneHelperTransportIdentity(device_id="d", transport="adb-tcp", transport_generation="x",
                                         minimum_usb_speed_mbps=1, hardware_identity={}, software_identity={},
                                         receipts={})

    def aoa_bundle(self, directory: str, *, relay_sha256: str = SHA) -> tuple[Path, dict]:
        path, value = static_campaign.StaticHelperCampaignTests.bundle(None, directory)
        bridge_script = Path(directory) / "aoa_bridge.py"
        bridge_script.write_text("# bridge fixture\n")
        aoa = AoaBridgeConfiguration.from_json({**AOA, "bridge_script_sha256": digest(bridge_script)})
        identity = value["transport_identity"]
        identity.update(transport="aoa-bridge", transport_generation=AOA_BRIDGE_TRANSPORT_GENERATION)
        identity["hardware_identity"]["aoa_usb_identity"] = "18d1:2d01"
        identity["software_identity"].update(phone_relay_sha256=relay_sha256, host_bridge_sha256=digest(bridge_script),
                                             aoa_bridge_options_sha256=aoa.options_sha256)
        receipt = value["receipt_paths"]["usb-link-speed"]
        kinds = IDENTITY_REQUIREMENTS["aoa-bridge"]["receipts"]
        identity["receipts"] = {kind: digest(receipt) for kind in kinds}
        value["receipt_paths"] = {kind: receipt for kind in kinds}
        value["host_software_paths"]["host_bridge_sha256"] = str(bridge_script)
        value["aoa_bridge"] = aoa.to_json()
        path.write_text(json.dumps(value))
        return path, value

    def test_plain_evidence_is_unchanged_and_builds_the_adb_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, value = static_campaign.StaticHelperCampaignTests.bundle(None, directory)
            evidence = load_helper_evidence(path)
            self.assertIsNone(evidence.aoa_bridge)
            self.assertIs(type(evidence.session()), AdbTcpPhoneWorkerSession)
            value["aoa_bridge"] = AOA
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(PhysicalAdapterError, "only aoa-bridge"):
                load_helper_evidence(path)

    def test_aoa_evidence_pins_relay_bridge_and_options_and_builds_the_aoa_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, value = self.aoa_bundle(directory)
            evidence = load_helper_evidence(path)
            self.assertEqual(evidence.aoa_bridge.relay_sha256, SHA)
            session = evidence.session()
            self.assertIsInstance(session, AoaBridgePhoneWorkerSession)
            self.assertEqual(session.transport_parameters.__func__.__qualname__,
                             "AoaBridgePhoneWorkerSession.transport_parameters")
            # the relay pin must agree with the bundle's software identity
            path, value = self.aoa_bundle(directory, relay_sha256="sha256:" + "c" * 64)
            with self.assertRaisesRegex(PhysicalAdapterError, "AOA bridge differs"):
                load_helper_evidence(path)
            path, value = self.aoa_bundle(directory)
            (Path(directory) / "aoa_bridge.py").write_text("# edited\n")
            with self.assertRaisesRegex(PhysicalAdapterError, "host software differs"):
                load_helper_evidence(path)
            path, value = self.aoa_bundle(directory)
            del value["aoa_bridge"]
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(PhysicalAdapterError, "lacks its bridge"):
                load_helper_evidence(path)

    def test_declaration_check_binds_rig_row_transport_and_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, value = self.aoa_bundle(directory)
            evidence = load_helper_evidence(path)
            model = SimpleNamespace(artifact_sha256=h.EVIDENCE, embedding_length=32, feed_forward_length=128)
            worker = evidence.worker
            row = SimpleNamespace(
                device_id=worker.device_id, serial=worker.serial, adb_port=5037, backend="CPU",
                worker_path=worker.worker_path, library_directories=worker.library_directories,
                column_quantum=32, max_tokens=4, forward_port=26991, max_requests=0,
                worker_environment=worker.worker_environment, as_root=False, phone_lock_path=None,
                worker_port=26990, transport="aoa-bridge", aoa_bridge=value["aoa_bridge"])
            helper = SimpleNamespace(device_id=worker.device_id, layer_mask=h.PIXEL_MASK, shard_sha256=h.SHARD_SHA,
                                     transport_parameters={**h.PIXEL_TRANSPORT, "ffn_link_transport": "aoa-bridge"})
            declaration = SimpleNamespace(helpers=(helper,))
            validate_helper_declaration(declaration, model, evidence, row)
            for change, parameters in (({"transport": "adb-tcp"}, None),
                                       ({"aoa_bridge": {**value["aoa_bridge"], "bridge_options": {}}}, None),
                                       ({}, dict(h.PIXEL_TRANSPORT))):
                with self.subTest(change=change):
                    bad_row = SimpleNamespace(**{**vars(row), **change})
                    bad_helper = SimpleNamespace(**{**vars(helper), **(
                        {"transport_parameters": parameters} if parameters is not None else {})})
                    with self.assertRaisesRegex(PhysicalAdapterError, "differs from its evidence"):
                        validate_helper_declaration(SimpleNamespace(helpers=(bad_helper,)), model, evidence, bad_row)


class PrepareAoaEvidenceTests(unittest.TestCase):
    """Phase-B tooling: the qualified adb-tcp bundle + A/B receipts + server identity -> aoa-bridge bundle/arm."""

    def fixture(self, directory: str) -> argparse.Namespace:
        from research_dev.scheduler.campaigns.burstgpt.tools.pixel_transport_ab import bridge_configuration
        root = Path(directory)
        adb_path, value = static_campaign.StaticHelperCampaignTests.bundle(None, directory)
        script = root / "aoa_bridge.py"
        script.write_text("# deployed bridge\n")
        aoa = {**AOA, "bridge_script_sha256": digest(script)}
        (root / "AOA_BRIDGE.json").write_text(json.dumps(aoa))
        ab = root / "ab"
        run = ab / "02-aoa"
        run.mkdir(parents=True)
        (ab / "CONFIG.json").write_text(json.dumps({"aoa_bridge": aoa}))
        arm = {"name": "aoa", "transport": "aoa-bridge"}
        (run / "ARM.json").write_text(json.dumps(arm))
        self.assertEqual(bridge_configuration({"aoa_bridge": aoa}, arm).options_sha256,
                         AoaBridgeConfiguration.from_json(aoa).options_sha256)
        (run / "PREFLIGHT.json").write_text(json.dumps({"aoa_bridge": {"relay_sha256": SHA,
                                                                       "bridge_script_sha256": digest(script)}}))
        (run / "HOST_USB_BEFORE.json").write_text(json.dumps({"device": {
            "serial": h.PIXEL_SERIAL, "vendor_product": "18d1:2d01", "sysfs_device": "2-9.2", "speed_mbps": 5000},
            "lpm": {"usb3_hardware_lpm_u2": "enabled"}, "usbfs_memory_mb": "16"}))
        (run / "RESULT.json").write_text('{"status": "PASS"}')
        (ab / "aoa-bridge-round-trip.json").write_text(json.dumps({
            "status": "PASS", "runs": ["02-aoa"], "calibration_arm": "aoa", "effective_transfer_bytes_per_s": 150000000,
            "one_direction_fixed_us": 250, "overhead_us_by_rows": {"1": 900, "2": 1100, "4": 1400}}))
        (ab / "aoa-bridge-byte-identity.json").write_text('{"status": "PASS"}')
        (ab / "aoa-bridge-scheduler-launched-session.json").write_text('{"status": "PASS"}')
        server = root / "server-identity-aoa"
        server.mkdir()
        (server / "RESULT.json").write_text(json.dumps({
            "status": "PASS", "token_identity": [True] * 4, "transport": "aoa-bridge", "server_exit": 0,
            "worker_exit": 0, "phone_calls": 12, "boot_unchanged": True, "output_tokens_each": 64,
            "call_columns": {"8704": 6, "17408": 6}}))
        worker = value["worker"]
        (server / "CONFIG.json").write_text(json.dumps({
            "phone_worker": worker["worker_path"], "phone_model": worker["shard_path"],
            "phone_library_dir": worker["library_directories"][0], "phone_environment": {}}))
        configuration = AoaBridgeConfiguration.from_json(aoa)
        (server / "IDENTITY.json").write_text(json.dumps({
            "phone_hashes": "".join(f"{pin[7:]}  {path}\n" for path, pin in worker["expected_sha256_by_path"].items()),
            "server_sha256": digest(root / "server")[7:],
            "aoa_bridge": {"options_sha256": configuration.options_sha256,
                           "configuration": configuration.to_json()}}))
        return argparse.Namespace(adb_evidence=str(adb_path), ab_dir=str(ab), server_identity=str(server),
                                  aoa_bridge=str(root / "AOA_BRIDGE.json"), bridge_script=str(script),
                                  output_dir=str(root / "aoa-evidence"))

    def test_bundle_keeps_the_worker_and_replaces_the_transport(self) -> None:
        from research_dev.scheduler.campaigns.burstgpt.tools import prepare_aoa_evidence as tool
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            path = tool.build_bundle(args)
            evidence = load_helper_evidence(path)
            old = load_helper_evidence(args.adb_evidence)
            self.assertEqual(evidence.identity.transport, "aoa-bridge")
            self.assertTrue(evidence.identity.qualified)
            self.assertEqual(evidence.worker, old.worker)
            self.assertEqual(evidence.identity.software_identity["phone_worker_sha256"],
                             old.identity.software_identity["phone_worker_sha256"])
            self.assertEqual(evidence.identity.receipts["numerical-rows-1-2-4"],
                             old.identity.receipts["numerical-rows-1-2-4"])
            links = {row["link_id"]: row for row in evidence.profile_fragment["links"]}
            self.assertEqual(set(links), {"pixel-aoa-out", "pixel-aoa-in"})
            self.assertEqual((links["pixel-aoa-out"]["source_device"], links["pixel-aoa-out"]["target_device"],
                              links["pixel-aoa-out"]["fixed_latency_us"]), ("desk-cpu", h.PIXEL, 250))
            self.assertIsInstance(evidence.session(), AoaBridgePhoneWorkerSession)

    def test_bundle_refuses_mismatched_inputs(self) -> None:
        from research_dev.scheduler.campaigns.burstgpt.tools import prepare_aoa_evidence as tool
        cases = (
            ("server", lambda root: (root / "server-identity-aoa" / "RESULT.json").write_text(json.dumps({
                **json.loads((root / "server-identity-aoa" / "RESULT.json").read_text()),
                "token_identity": [True, False, True, True]})), "server identity over the bridge failed"),
            ("options", lambda root: (root / "ab" / "02-aoa" / "ARM.json").write_text(json.dumps({
                "name": "aoa", "transport": "aoa-bridge", "relay_options": {"qos_latency_us": 5}})),
             "other keep-awake options"),
            ("usb", lambda root: (root / "ab" / "02-aoa" / "HOST_USB_BEFORE.json").write_text(json.dumps({
                "device": {"serial": h.PIXEL_SERIAL, "vendor_product": "18d1:2d00", "sysfs_device": "2-9.2",
                           "speed_mbps": 5000}})), "accessory-mode USB link"),
            ("script", lambda root: (root / "aoa_bridge.py").write_text("# other\n"), "deployed host bridge"),
        )
        for name, mutate, message in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(directory)
                mutate(Path(directory))
                with self.assertRaisesRegex(PhysicalAdapterError, message):
                    tool.build_bundle(args)

    def test_adb_link_calibration_keeps_the_old_link_rows(self) -> None:
        from research_dev.scheduler.campaigns.burstgpt.tools import prepare_aoa_evidence as tool
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            args.link_calibration = "adb"
            evidence = load_helper_evidence(tool.build_bundle(args))
            old = load_helper_evidence(args.adb_evidence)
            self.assertEqual(evidence.profile_fragment["links"], old.profile_fragment["links"])

    def test_server_identity_tool_maps_the_derived_config_to_the_worker(self) -> None:
        from research_dev.scheduler.campaigns.burstgpt.tools.qualify_pixel_server_transport import worker_configuration
        config = {"serial": h.PIXEL_SERIAL, "phone_worker": "/p/worker", "phone_model": "/p/shard.gguf",
                  "phone_library_dir": "/p", "phone_environment": {"S43_PIXEL_CPU_POLL": "100"}, "phone_root": True,
                  "phone_backend": "CPU", "phone_port": 27191, "output_tokens": 64,
                  "expected_phone_sha256": {"/p/worker": "a" * 64, "/p/shard.gguf": "b" * 64},
                  "phone": {"artifact_sha256": SHA, "layers": [18, 19, 20, 21, 22, 23], "n_embd": 5120,
                            "columns": 17408, "quantum": 4352}}
        worker = worker_configuration(config, 26991)
        self.assertEqual((worker.max_requests, worker.forward_port, worker.layer_mask, worker.as_root),
                         (64 * 2 * 6, 26991, 0xFC0000, True))
        self.assertEqual(worker.expected_sha256_by_path["/p/worker"], "sha256:" + "a" * 64)
        self.assertEqual(worker.phone_lock_path, "/data/local/tmp/.s42-pixel-ffn-kernels.lock")

    def test_derived_arm_changes_only_the_pixel_transport(self) -> None:
        from research_dev.scheduler.campaigns.burstgpt.tools import prepare_aoa_evidence as tool
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = tool.build_bundle(self.fixture(directory))
            inputs = root / "inputs-two-phone-x"
            inputs.mkdir()
            rig = json.loads(json.dumps(two_phone_rig_json()).replace("pixel10pro-phone", h.PIXEL))
            rig["helper_phones"][0]["forward_port"] = 26991
            (inputs / "rig.json").write_text(json.dumps(rig))
            (inputs / "campaign.json").write_text(json.dumps({
                "campaign_id": "s43-two-phone-eval-x", "rig_manifest_path": str(inputs / "rig.json"),
                "evidence_manifest_path": str(inputs / "evidence.json"), "models_manifest_path": str(inputs / "models.json")}))
            (inputs / "evidence.json").write_text(json.dumps({"helper_phone_evidence_paths": {h.PIXEL: "/old.json"},
                                                              "kernel_profile_path": "/k.json"}))
            (inputs / "models.json").write_text("{}")
            target = tool.derive_arm(argparse.Namespace(inputs=str(inputs), evidence=str(evidence),
                                                         output=str(root / "inputs-two-phone-x-aoa")))
            derived = json.loads((target / "rig.json").read_text())
            original = json.loads((inputs / "rig.json").read_text())
            helper = derived["helper_phones"][0]
            self.assertEqual((helper["transport"], helper["aoa_bridge"]["relay_sha256"]), ("aoa-bridge", SHA))
            del helper["transport"], helper["aoa_bridge"]
            del original["helper_phones"][0]["transport"]
            self.assertEqual(derived, original)
            campaign = json.loads((target / "campaign.json").read_text())
            self.assertEqual(campaign["campaign_id"], "s43-two-phone-eval-x-aoa")
            self.assertEqual(campaign["rig_manifest_path"], str(target.resolve() / "rig.json"))
            manifest = json.loads((target / "evidence.json").read_text())
            self.assertEqual(manifest["helper_phone_evidence_paths"][h.PIXEL], str(evidence.resolve()))
            self.assertEqual(manifest["kernel_profile_path"], "/k.json")


if __name__ == "__main__":
    unittest.main()
