"""Reproduce the failed ticket and validate its RPC binding without hardware execution."""

import dataclasses
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
from research_dev.scheduler import RuntimeCapabilityCatalog
from research_dev.scheduler._internal.route_generation.costing_parameters import RouteParameterMixin
from research_dev.scheduler._internal.route_generation.feasibility import RouteFeasibilityMixin
from research_dev.scheduler._internal.route_generation.remote_resident import remote_resident_link_ids
from research_dev.scheduler._internal.types import canonical_json
from research_dev.scheduler.adapters.phone_transport import phone_transport_contract

source = ROOT / "gate-run-v2/phone"
ticket = json.loads((source / "snapshots/reduced-00-42-ticket.json").read_text())
parameters = dict(ticket["execution_plan"]["adapter_parameters"])
try:
    phone_transport_contract(parameters)
except Exception as error:
    before_error = str(error)
else:
    raise AssertionError("the saved failure was not reproduced")
catalog = RuntimeCapabilityCatalog.from_json(json.loads((ROOT / "GATE_CATALOG-v2.json").read_text()))
links = remote_resident_link_ids(catalog.placement_profile, parameters)
assert links and all("link:" + key in catalog.resources for key in links)
compiler = SimpleNamespace(_transport_adapter_parameters=RouteFeasibilityMixin._transport_adapter_parameters)
reason = RouteParameterMixin._candidate_transport_parameters(
    compiler, catalog.placement_profile, links,
    SimpleNamespace(input_tokens=ticket["request"]["input_tokens"], output_tokens=ticket["request"]["output_tokens"]),
    SimpleNamespace(assistance_phase="all"), parameters)
assert reason is None, reason
contract = phone_transport_contract(parameters)
preparation = json.loads((source / "PREPARATION.json").read_text())
loaded = phone_transport_contract(preparation["commands"][0]["adapter_parameters"])
assert contract.shares_resident_session_with(loaded)
assert contract.max_payload_bytes == loaded.max_payload_bytes
value = {
    "status": "PASS", "physical_execution": False, "before_error": before_error,
    "source_ticket": str(source / "snapshots/reduced-00-42-ticket.json"),
    "required_link_resources": ["link:" + key for key in links],
    "new_transport": dataclasses.asdict(contract), "loaded_transport": dataclasses.asdict(loaded),
    "resident_router_reusable": True,
}
with (ROOT / "TRANSPORT_BINDING_REPLAY.json").open("x") as stream:
    stream.write(canonical_json(value) + "\n")
print(canonical_json(value))
