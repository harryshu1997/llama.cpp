"""M3 mechanism gate: one llama-server, OP15 FunctionFS sessions plus a second phone's TCP worker.

Runs the decode-relocation gate (``reports/20260917-decode-relocation-kv-headroom/
kv_decode_relocation_gate.py``, the M0/M2 harness) unchanged, with its phone owner extended by one
helper phone from ``config["helper_phone"]``:

* the OP15 sessions keep their layers (``config["phone"]["session_masks"]``); the helper phone owns
  the disjoint ``helper_phone.layer_mask`` through a direct protocol-v6 worker behind
  ``adb forward`` (:mod:`adapters.phone_tcp_session`);
* the server is launched with ``S41_SERVER_FFN_HELPERS=2`` and the union layer mask, and the
  gate's decode policy covers both ranges with one column width;
* per-request proofs gain a ``PIXEL0``-style shard row, so every call must reach its owner;
* after the run the helper's finite request budget is drained with zero-input calls (never a
  signal) and ``TWO_PHONE_RESULT.json`` adds per-device call accounting and the helper's
  assumed power, kept separate from the measured host energy and from OP15's assumption.

Arms (M3 check): ``--arm control`` (desktop only), ``--arm combined --host-columns 0
--without-helper-phone`` (OP15 alone, 100 % of its layers), ``--arm combined --host-columns H``
with the helper at the same total assisted columns x layers, and ``--host-columns 0`` with the
helper as the capacity arm. All gate options apply; the rig lock is the gate's.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_dev.scheduler._internal.plan_contracts.phone import RuntimePhoneShard  # noqa: E402
from research_dev.scheduler._internal.types import canonical_sha256  # noqa: E402
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError  # noqa: E402
from research_dev.scheduler.adapters.llama_server_contracts import parse_llama_server_ffn_call  # noqa: E402
from research_dev.scheduler.adapters.phone_helpers import (  # noqa: E402
    PhoneHelperBinding,
    attribute_ffn_calls,
    helper_server_environment,
)
from research_dev.scheduler.adapters.phone_tcp_session import (  # noqa: E402
    AdbTcpPhoneWorkerSession,
    AdbTcpWorkerConfiguration,
)

GATE_PATH = Path(__file__).resolve().parent / (
    "reports/20260917-decode-relocation-kv-headroom/kv_decode_relocation_gate.py")
HELPER_SESSION_ID = "PIXEL0"
_TRANSPORT_KEYS = ("S41_SERVER_FFN_TRANSPORT", "S41_SERVER_FFN_HOST", "S41_SERVER_FFN_PORT")


def load_gate():
    spec = importlib.util.spec_from_file_location("kv_decode_relocation_gate", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Receipt:
    def __init__(self, value: dict) -> None:
        self.value = value

    def to_json(self) -> dict:
        return self.value


def helper_configuration(config: dict, artifact_sha256: str, n_embd: int, swiglu: bool) -> AdbTcpWorkerConfiguration:
    row = config["helper_phone"]
    return AdbTcpWorkerConfiguration(
        device_id=row["device_id"], serial=row["serial"], adb_port=int(row.get("adb_port", 5037)),
        adb_path=Path(row.get("adb_path", "/usr/bin/adb")), worker_path=row["worker_path"],
        library_directories=tuple(row["library_directories"]), shard_path=row["shard_path"],
        artifact_sha256=artifact_sha256, layer_mask=int(row["layer_mask"]), n_embd=n_embd,
        columns=int(row["columns"]), column_quantum=int(row["column_quantum"]),
        max_tokens=int(row["max_tokens"]), swiglu=swiglu, backend=row["backend"],
        phone_port=int(row["phone_port"]), forward_port=int(row.get("forward_port", 0)),
        max_requests=int(row["max_requests"]), worker_environment=row.get("worker_environment", {}),
        expected_sha256_by_path=row["expected_sha256_by_path"],
        as_root=row.get("as_root", False), phone_lock_path=row.get("phone_lock_path"),
    )


class TwoPhoneOwner:
    """The gate's phone owner for two phones: OP15 sessions first, then the helper worker."""

    def __init__(self, op15, helper: AdbTcpPhoneWorkerSession, label: str, op15_serial: str, op15_mask: int,
                 environment: dict, output: Path) -> None:
        self.op15, self.helper, self.label = op15, helper, label
        self.op15_binding = PhoneHelperBinding("op15-phone", op15_serial, op15_mask, "op15")
        self.environment, self.output = environment, output
        self.helper_proofs: dict[str, list[dict]] = {}
        self.helper_started = False

    def preflight(self):
        return _Receipt({"op15": self.op15.preflight().to_json(), "helper": self.helper.preflight().to_json()})

    def start(self, command, manifest, usb):
        ready = self.op15.start(command, manifest, usb)
        launch = self.helper.start(self.output / "helper-worker.log")
        self.helper_started = True
        shared = {key: value for key, value in self.environment.items()
                  if key not in _TRANSPORT_KEYS and not key.startswith(("S41_SERVER_FFN_USB_", "S41_SERVER_FFN_USBFS_"))}
        helpers = (self.op15_binding, self.helper.binding(self.label))
        composed = helper_server_environment(helpers, shared, (usb, self.helper.transport_contract()))
        # the gate launches the server from this same dict after the owner started
        self.environment.clear()
        self.environment.update(composed)
        return _Receipt({"op15": ready.to_json(), "helper": launch.to_json(),
                         "helpers": [row.to_json() for row in helpers]})

    def bind_ticket_generation(self, ticket_id: str) -> int:
        return self.op15.bind_ticket_generation(ticket_id)

    def record_execution_proof(self, ticket_id: str, artifact_sha256: str, proofs) -> None:
        own = tuple(row for row in proofs if row.session_id != HELPER_SESSION_ID)
        self.helper_proofs[ticket_id] = [row.to_json() for row in proofs if row.session_id == HELPER_SESSION_ID]
        if own:
            self.op15.record_execution_proof(ticket_id, artifact_sha256, own)

    def helper_calls(self) -> int:
        """Calls the server made on the helper's layers (its request-id range)."""
        lines = json.loads((self.output / "SERVER_FFN_LINES.json").read_text())
        calls = [call for call in (parse_llama_server_ffn_call(line) for line in lines) if call is not None]
        helper = self.helper.binding(self.label)
        return attribute_ffn_calls(calls, (self.op15_binding, helper))[helper.device_id].calls

    def finish(self, *, require_execution: bool):
        close = self.op15.finish(require_execution=require_execution)
        served = self.helper_calls()
        stop = self.helper.stop(served_calls=served)
        self.helper_started = False
        return _Receipt({"op15": close.to_json(), "helper": stop.to_json(), "helper_served_calls": served,
                         "helper_proofs": self.helper_proofs})

    def abort(self):
        receipt = {"op15": self.op15.abort().to_json()}
        if self.helper_started:
            try:
                served = self.helper_calls() if (self.output / "SERVER_FFN_LINES.json").exists() else None
                receipt["helper"] = self.helper.stop(served_calls=served).to_json()
            except (OSError, ValueError, PhysicalAdapterError) as error:
                # an in-flight or undrained worker is left running, never killed
                receipt["helper_left_running"] = repr(error)
        return _Receipt(receipt)


def install(gate, *, with_helper: bool):
    """Wrap the gate's phone owner and run so its combined arms drive both phones."""
    original_owner, original_run = gate.phone_owner, gate.run

    def phone_owner(config, manifest, output, plan, fractions_ppm, dormant):
        owner, command, usb, env, mask, shards = original_owner(config, manifest, output, plan, fractions_ppm, dormant)
        if not with_helper:
            return owner, command, usb, env, mask, shards
        row = config["helper_phone"]
        helper_mask = int(row["layer_mask"])
        first_gpu_layer = max(0, manifest.block_count + 1 - config["gpu_layers"])
        if helper_mask & mask or helper_mask >> first_gpu_layer:
            raise ValueError("helper phone layers must be CPU-parent layers disjoint from the OP15 sessions")
        if int(row["columns"]) != manifest.feed_forward_length or int(row["column_quantum"]) % 32:
            raise ValueError("helper phone must hold the complete FFN width on a 32-column quantum")
        if str(row["max_tokens"]) != env["S41_SERVER_FFN_MAX_TOKENS"]:
            raise ValueError("helper phone maximum tokens must equal the server's (HELLO compares them)")
        swiglu = env["S41_SERVER_FFN_ACTIVATION"] == "swiglu"
        helper = AdbTcpPhoneWorkerSession(helper_configuration(
            config, manifest.artifact_sha256, manifest.embedding_length, swiglu))
        weight_bytes = sum(manifest.tensor_by_id[f"blk.{il}.ffn_{kind}.weight"].nbytes
                           for il in range(manifest.block_count) if helper_mask >> il & 1
                           for kind in ("gate", "up", "down"))
        shard = RuntimePhoneShard(
            HELPER_SESSION_ID, f"adb-tcp://{row['serial']}/{HELPER_SESSION_ID}", helper_mask,
            manifest.feed_forward_length, weight_bytes,
            canonical_sha256({"artifact": manifest.artifact_sha256, "mask": helper_mask,
                              "columns": manifest.feed_forward_length}),
            plan.plan_sha256, manifest.artifact_sha256, 1)
        env["S41_SERVER_FFN_LAYER_MASK"] = str(mask | helper_mask)
        owner = TwoPhoneOwner(owner, helper, row.get("label", "pixel"), config["phone"]["serial"], mask, env, output)
        return owner, command, usb, env, mask | helper_mask, (*shards, shard)

    def run(config, output, arm, options):
        original_run(config, output, arm, options)
        if with_helper and arm != "control" and (output / "PHONE_CLOSE.json").exists():
            write_two_phone_result(config, output)

    gate.phone_owner, gate.run = phone_owner, run
    return gate


def write_two_phone_result(config: dict, output: Path) -> dict:
    """Per-device call accounting and separately assumed helper energy for one completed arm."""
    row = config["helper_phone"]
    result = json.loads((output / "RESULT.json").read_text())
    close = json.loads((output / "PHONE_CLOSE.json").read_text())
    op15_mask = sum(int(value) for value in config["phone"]["session_masks"].values())
    bindings = (PhoneHelperBinding("op15-phone", config["phone"]["serial"], op15_mask, "op15"),
                PhoneHelperBinding(row["device_id"], row["serial"], int(row["layer_mask"]), row.get("label", "pixel")))
    lines = json.loads((output / "SERVER_FFN_LINES.json").read_text())
    calls = [call for call in (parse_llama_server_ffn_call(line) for line in lines) if call is not None]
    accounts = attribute_ffn_calls(calls, bindings)
    decode_s = sum(request.get("decode_s") or 0.0 for request in result["requests"] if request.get("split"))
    power = row.get("assumed_power_w", {"idle": 0.875, "active": 4.5})
    summary = {
        "schema": "s42-two-phone-mechanism-gate-v1",
        "status": result.get("status"),
        "calls_by_device": {key: value.to_json() for key, value in accounts.items()},
        "helper_served_calls": close.get("helper_served_calls"),
        "helper_stop": close.get("helper"),
        "helper_assumed_j": {
            "evidence": "ASSUMED, not measured; separate from host RAPL/NVML and from OP15's assumed power",
            "idle_w": power["idle"], "active_w": power["active"],
            "assisting": decode_s * power["active"] + (result["paid_s"] - decode_s) * power["idle"],
        },
        "host_energy": result.get("paid_host_energy"),
        "phone_proof_summary": result.get("phone_proof_summary"),
        "dormant_proofs": result.get("dormant_proofs"),
    }
    (output / "TWO_PHONE_RESULT.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


def main() -> None:
    with_helper = "--without-helper-phone" not in sys.argv
    if not with_helper:
        sys.argv.remove("--without-helper-phone")
    install(load_gate(), with_helper=with_helper).main()


if __name__ == "__main__":
    main()
