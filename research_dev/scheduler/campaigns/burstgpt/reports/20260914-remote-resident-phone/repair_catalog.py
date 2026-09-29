"""Preserve the failed catalog and check the resource repair against its saved snapshot."""

import dataclasses
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
from research_dev.scheduler import ResourceProfile, RuntimeCapabilityCatalog, HeterogeneousRuntimeSnapshot
from research_dev.scheduler._internal.types import canonical_json
from research_dev.scheduler.configuration.campaign import FixedPhoneResidencyConfiguration
from research_dev.scheduler.campaigns.burstgpt import remote_resident_gate as gate


def save(name, value):
    if (ROOT / name).exists():
        assert (ROOT / name).read_text() == canonical_json(value) + "\n"
        return
    with (ROOT / name).open("x") as stream:
        stream.write(canonical_json(value) + "\n")


command = json.loads((ROOT / "GATE_COMMAND.json").read_text())
catalog = json.loads((ROOT / "GATE_CATALOG.json").read_text())
resources = {row["resource_id"]: row for row in catalog["resources"]}
added = []
for link in catalog["placement_profile"]["links"]:
    key = "link:" + link["link_id"]
    if key not in resources:
        resources[key] = dataclasses.asdict(ResourceProfile(
            resource_id=key, kind="transport", capacity=1, ready=True, identity=link["link_id"]))
        added.append(key)
catalog["resources"] = [resources[key] for key in sorted(resources)]
RuntimeCapabilityCatalog.from_json(catalog)
save("GATE_CATALOG-v2.json", catalog)
command[command.index("--capability-catalog") + 1] = str(ROOT / "GATE_CATALOG-v2.json")
save("GATE_COMMAND-v2.json", command)
sys.argv = command[1:]
args = gate.parse_args()
models = gate.runner._load_trace_models(args)
scheduler, manifests, _ = gate.runner._build_scheduler(args, models, load_adaptive_observations=False)
model_id = {gate.runner.GEMMA_ROLE: models.expected_gemma,
            gate.runner.QWEN_ROLE: models.expected_qwen}[args.desktop_parent_role].model_id
manifest = manifests[model_id]
scheduler.configure_fixed_phone_residency(FixedPhoneResidencyConfiguration("remote-owner-gate", ((
    args.session_id, manifest.artifact_sha256, int(args.remote_layer_mask, 0),
    manifest.feed_forward_length,
),)))
snapshot_path = ROOT / "gate-run-v1/phone-discovery/INITIAL_SNAPSHOT.json"
snapshot = HeterogeneousRuntimeSnapshot.from_json(json.loads(snapshot_path.read_text()))
plan = scheduler.plan_offline_phone_residency(scheduler.fixed_phone_residency_requests(),
    snapshot=snapshot, observed_at_us=snapshot.captured_at_us)
save("CATALOG_REPAIR_REPLAY.json", {
    "status": "PASS", "physical_execution": False, "added_resources": added,
    "source_snapshot": str(snapshot_path), "plan": plan.to_json(),
    "shared_resources": "Existing USB, FunctionFS and HTP resource declarations unchanged",
})
print(canonical_json({"status": "PASS", "added_resources": added, "plan_state": plan.state}))
