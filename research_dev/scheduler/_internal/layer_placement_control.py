"""Rebalancing of measured layer placement on events, with hysteresis that pays for every change (WS11).

`LayerPlacementController` keeps, per model, the ownership the next server launch will use (``target``),
the ownership of the running server (``launched``) and what is effectively executing (``current``: the
launched ownership minus helpers masked out at runtime). On every event -- device join / loss /
quarantine / readmission, thermal exclusion and clearance, a model added or removed, a server launch or
exit, new measured call summaries (drift) or new demand -- it re-plans with
:func:`layer_placement.solve_placement` and classifies each difference:

``MASK_OUT_NOW``      mandatory: a layer's owner became unavailable; the host takes it at runtime
                      (llama-server's runtime FFN control narrows the union mask; no restart, no cost).
``ADOPT_RUNTIME``     the new ownership is inside what the running server launched with (a narrowing or
                      the restoration of a masked-out helper): runtime layer-mask control, free.
``ADOPT_NEXT_LAUNCH`` a new owner for some layer: llama-server binds helper ownership at launch, so the
                      change waits for the model's next server launch (free when it is not running).
``RESTART_NOW``       the gain over the remaining horizon pays for relaunching the running server.
``PROVISION``         the ideal plan needs shards a device does not store (and evidence covers them): push
                      them (phone residency provisioning); adopted once the device reports them stored.
``BLOCKED_QUALIFICATION`` the ideal plan needs layers no PASS evidence covers: recorded, never executed.
``HOLD``              a better plan exists but its gain over the horizon does not beat the switching cost
                      by the margin, or the model changed placement less than ``min_dwell_s`` ago.

Only measured improvements move layers: drift re-plans only when a cost the current plan relies on
moved by more than ``drift_ppm``. Pure state machine, no I/O, deterministic for a given event stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping, Sequence

from .layer_placement import (
    DESKTOP_CPU,
    HelperDevice,
    LayerPlacementError,
    LayerPlacementProfile,
    ModelLayers,
    PlacementProblem,
    evaluate_owners,
    layer_mask,
    layer_spec,
    mask_layers,
    provisioning_needs,
    restrict_to_executable,
    solve_placement,
    with_envelope_caps,
)


LAYER_PLACEMENT_CONTROL_SCHEMA = "research-scheduler-layer-placement-control-v1"
EVENT_KINDS = (
    "START", "DEVICE_JOINED", "DEVICE_LEFT", "DEVICE_QUARANTINED", "DEVICE_READMITTED", "THERMAL_EXCLUDED",
    "THERMAL_CLEARED", "MODEL_ADDED", "MODEL_REMOVED", "SERVER_LAUNCH", "SERVER_EXIT", "SHAPES_OBSERVED",
    "SHARDS_STORED", "DEMAND", "PERIODIC",
)
ACTIONS = ("KEEP", "MASK_OUT_NOW", "ADOPT_RUNTIME", "ADOPT_NEXT_LAUNCH", "RESTART_NOW", "PROVISION",
           "BLOCKED_QUALIFICATION", "HOLD")
_UNAVAILABLE_KINDS = {"DEVICE_LEFT": "LEFT", "DEVICE_QUARANTINED": "QUARANTINED", "THERMAL_EXCLUDED": "THERMAL"}
_AVAILABLE_KINDS = {"DEVICE_JOINED": "LEFT", "DEVICE_READMITTED": "QUARANTINED", "THERMAL_CLEARED": "THERMAL"}


@dataclass(frozen=True)
class RebalancePolicy:
    """Hysteresis and switching costs.

    A change is adopted when ``gain_j_per_token * horizon_tokens > cost_j * (1 + margin_ppm / 1e6)`` and
    the gain per token is at least ``min_gain_j_per_token``; optional changes also wait ``min_dwell_s``
    after the model's previous change. ``horizon_tokens`` = demand (tokens/s, from DEMAND events or
    ``default_tokens_per_s``) times ``horizon_s``. Costs: a shard push moves ``bytes / push_bytes_per_s``
    seconds at ``push_power_w``; a forced relaunch costs ``restart_j[model]`` (desktop reload energy).
    """

    horizon_s: float = 900.0
    margin_ppm: int = 250_000
    min_dwell_s: float = 120.0
    drift_ppm: int = 150_000
    min_gain_j_per_token: float = 0.05
    push_bytes_per_s: float = 200e6
    push_power_w: float = 60.0
    default_tokens_per_s: float = 1.0
    restart_j: Mapping[str, float] = MappingProxyType({})
    # safe exploration: a device's busy time per step may exceed its measured sustained envelope by at
    # most this much (None = no envelope cap); the envelope grows as sustained operation is measured
    busy_growth_ppm: int | None = None
    sustained_calls: int = 2000
    # risk premiums by provenance (see PlacementProblem.uncertainty_ppm) and the confident sample size
    uncertainty_ppm: Mapping[str, int] = MappingProxyType({})
    confident_calls: int = 0

    def __post_init__(self) -> None:
        for name in ("horizon_s", "min_dwell_s", "min_gain_j_per_token", "push_power_w", "default_tokens_per_s"):
            if not isinstance(getattr(self, name), (int, float)) or getattr(self, name) < 0:
                raise LayerPlacementError("rebalance " + name + " must be >= 0")
        if self.push_bytes_per_s <= 0:
            raise LayerPlacementError("rebalance push_bytes_per_s must be > 0")
        for name in ("margin_ppm", "drift_ppm", "sustained_calls", "confident_calls"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise LayerPlacementError("rebalance " + name + " must be an integer >= 0")
        if self.busy_growth_ppm is not None and (type(self.busy_growth_ppm) is not int or self.busy_growth_ppm < 0):
            raise LayerPlacementError("rebalance busy_growth_ppm must be an integer >= 0 or None")

    def adopts(self, gain_j_per_token: float, tokens: float, cost_j: float) -> bool:
        return (gain_j_per_token >= self.min_gain_j_per_token
                and gain_j_per_token * tokens > cost_j * (1 + self.margin_ppm / 1e6))

    def to_json(self) -> dict[str, object]:
        return {"horizon_s": self.horizon_s, "margin_ppm": self.margin_ppm, "min_dwell_s": self.min_dwell_s,
                "drift_ppm": self.drift_ppm, "min_gain_j_per_token": self.min_gain_j_per_token,
                "push_bytes_per_s": self.push_bytes_per_s, "push_power_w": self.push_power_w,
                "default_tokens_per_s": self.default_tokens_per_s, "restart_j": dict(sorted(self.restart_j.items())),
                "busy_growth_ppm": self.busy_growth_ppm, "sustained_calls": self.sustained_calls,
                "uncertainty_ppm": dict(sorted(self.uncertainty_ppm.items())), "confident_calls": self.confident_calls}


@dataclass(frozen=True)
class PlacementEvent:
    kind: str
    at_s: float
    device_id: str | None = None
    model_id: str | None = None
    details: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise LayerPlacementError("unknown placement event " + str(self.kind))
        if not isinstance(self.at_s, (int, float)) or self.at_s < 0:
            raise LayerPlacementError("placement event time must be >= 0")


@dataclass(frozen=True)
class PlacementDecision:
    at_s: float
    trigger: str
    model_id: str
    action: str
    owners_before: Mapping[str, int]       # device -> mask (the ownership the decision changes)
    owners_after: Mapping[str, int]
    gain_j_per_token: float
    horizon_tokens: float
    cost_j: float
    reason: str
    details: Mapping[str, object] = MappingProxyType({})

    def to_json(self) -> dict[str, object]:
        def owners(value):
            return {device: layer_spec(mask_layers(mask)) for device, mask in sorted(value.items()) if mask}
        return {"at_s": round(self.at_s, 6), "trigger": self.trigger, "model_id": self.model_id,
                "action": self.action, "before": owners(self.owners_before), "after": owners(self.owners_after),
                "gain_j_per_token": round(self.gain_j_per_token, 6), "horizon_tokens": round(self.horizon_tokens, 3),
                "cost_j": round(self.cost_j, 3), "reason": self.reason, "details": dict(self.details)}


def owners_to_masks(owners: Mapping[int, str]) -> dict[str, int]:
    masks: dict[str, int] = {}
    for layer, owner in owners.items():
        masks[owner] = masks.get(owner, 0) | (1 << layer)
    return masks


def masks_to_owners(model: ModelLayers, masks: Mapping[str, int]) -> dict[int, str]:
    owners = {layer: DESKTOP_CPU for layer in model.cpu_layers}
    for device, mask in masks.items():
        for layer in mask_layers(mask):
            if layer in owners:
                owners[layer] = device
    return owners


@dataclass
class LayerPlacementController:
    models: dict[str, ModelLayers]
    devices: dict[str, HelperDevice]
    profile: LayerPlacementProfile
    policy: RebalancePolicy = field(default_factory=RebalancePolicy)
    rows_mix: dict[str, Mapping[int, float]] = field(default_factory=dict)
    latency_ppm: int | None = None
    column_fractions: dict[str, tuple[float, ...]] = field(default_factory=dict)
    # state
    unavailable: dict[str, set[str]] = field(default_factory=dict)        # device -> reasons
    target: dict[str, dict[int, str]] = field(default_factory=dict)       # next launch ownership
    launched: dict[str, dict[int, str]] = field(default_factory=dict)     # running server ownership
    current: dict[str, dict[int, str]] = field(default_factory=dict)      # effective (after mask-outs)
    last_change_s: dict[str, float] = field(default_factory=dict)
    demand_tokens_per_s: dict[str, float] = field(default_factory=dict)
    planned_costs: dict[str, dict[str, float]] = field(default_factory=dict)   # model -> owner -> J/layer-token
    envelope_ms: dict[str, float] = field(default_factory=dict)          # device -> sustained busy per step
    recommended: dict[str, tuple] = field(default_factory=dict)          # model -> last needs recorded
    decisions: list[PlacementDecision] = field(default_factory=list)
    plans: list[dict[str, object]] = field(default_factory=list)
    errors: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def create(cls, models: Sequence[ModelLayers], devices: Sequence[HelperDevice], profile: LayerPlacementProfile,
               *, initial: Mapping[str, Mapping[int, str]] | None = None, **options) -> "LayerPlacementController":
        controller = cls({row.model_id: row for row in models}, {row.device_id: row for row in devices}, profile,
                         **options)
        for model in models:
            owners = dict((initial or {}).get(model.model_id, {}))
            owners = {layer: owners.get(layer, DESKTOP_CPU) for layer in model.cpu_layers}
            controller.target[model.model_id] = dict(owners)
            controller.current[model.model_id] = dict(owners)
        for device in devices:
            if not device.available:
                controller.unavailable.setdefault(device.device_id, set()).add(device.unavailable_reason or "ABSENT")
            if device.busy_envelope_ms is not None:
                controller.envelope_ms[device.device_id] = device.busy_envelope_ms
        return controller

    # -- problem construction ----------------------------------------------------------------------
    def problem(self, *, current: Mapping[str, Mapping[int, str]] | None = None) -> PlacementProblem:
        devices = []
        for device in self.devices.values():
            reasons = self.unavailable.get(device.device_id, set())
            device = replace(device, available=not reasons, unavailable_reason=",".join(sorted(reasons)) or None,
                             busy_envelope_ms=self.envelope_ms.get(device.device_id, device.busy_envelope_ms))
            devices.append(device)
        problem = PlacementProblem(
            models=tuple(self.models[name] for name in sorted(self.models)),
            devices=tuple(sorted(devices, key=lambda row: row.device_id)),
            profile=self.profile,
            rows_mix=MappingProxyType({k: MappingProxyType(dict(v)) for k, v in self.rows_mix.items()}),
            latency_ppm=self.latency_ppm,
            column_fractions=MappingProxyType(dict(self.column_fractions)),
            current=MappingProxyType({k: MappingProxyType(dict(v)) for k, v in (current or self.target).items()}),
            uncertainty_ppm=MappingProxyType(dict(self.policy.uncertainty_ppm)),
            confident_calls=self.policy.confident_calls,
        )
        if self.policy.busy_growth_ppm is not None:
            problem = with_envelope_caps(problem, self.policy.busy_growth_ppm)
        return problem

    def _available(self, device_id: str) -> bool:
        return device_id == DESKTOP_CPU or (device_id in self.devices and not self.unavailable.get(device_id))

    def _tokens(self, model_id: str) -> float:
        rate = self.demand_tokens_per_s.get(model_id, self.policy.default_tokens_per_s)
        return rate * self.policy.horizon_s

    # -- event handling ----------------------------------------------------------------------------
    def handle(self, event: PlacementEvent) -> list[PlacementDecision]:
        before = len(self.decisions)
        kind = event.kind
        affected = sorted(self.models)
        replan = True
        if kind in _UNAVAILABLE_KINDS:
            self._require_device(event)
            self.unavailable.setdefault(event.device_id, set()).add(_UNAVAILABLE_KINDS[kind])
        elif kind in _AVAILABLE_KINDS:
            self._require_device(event)
            reasons = self.unavailable.get(event.device_id, set())
            reasons.discard(_AVAILABLE_KINDS[kind])
            if kind == "DEVICE_JOINED":
                reasons.discard("ABSENT")
                if "device" in event.details:
                    self.devices[event.device_id] = event.details["device"]   # a new resource's spec
        elif kind == "MODEL_ADDED":
            model = event.details.get("model")
            if not isinstance(model, ModelLayers):
                raise LayerPlacementError("MODEL_ADDED needs details.model (ModelLayers)")
            self.models[model.model_id] = model
            owners = {layer: DESKTOP_CPU for layer in model.cpu_layers}
            self.target[model.model_id] = dict(owners)
            self.current[model.model_id] = dict(owners)
            if "rows_mix" in event.details:
                self.rows_mix[model.model_id] = dict(event.details["rows_mix"])
        elif kind == "MODEL_REMOVED":
            for table in (self.models, self.target, self.current, self.launched, self.last_change_s,
                          self.planned_costs, self.rows_mix, self.demand_tokens_per_s):
                table.pop(event.model_id, None)
            affected = sorted(self.models)
        elif kind == "SERVER_LAUNCH":
            self._require_model(event)
            owners = dict(self.target[event.model_id])
            owners = {layer: (owner if self._available(owner) else DESKTOP_CPU) for layer, owner in owners.items()}
            self.launched[event.model_id] = dict(owners)
            self.current[event.model_id] = dict(owners)
            affected, replan = [event.model_id], False
            self._decide(event, event.model_id, "launch")
        elif kind == "SERVER_EXIT":
            self._require_model(event)
            self.launched.pop(event.model_id, None)
            affected, replan = [event.model_id], False
        elif kind == "SHAPES_OBSERVED":
            rows = list(event.details.get("rows", ()))
            for row in rows:
                self.profile.observe_call_summary(**row)
            if event.model_id in self.models:
                self._grow_envelopes(event.model_id, rows)
            affected = [name for name in sorted(self.models) if self._drifted(name)]
            replan = bool(affected)
        elif kind == "SHARDS_STORED":
            self._require_device(event)
            layers = int(event.details.get("layer_mask", 0))
            qualified = int(event.details.get("qualified_mask", 0))
            device = self.devices[event.device_id]
            stored = dict(device.stored_layers)
            stored[event.model_id] = stored.get(event.model_id, 0) | layers
            evidence = dict(device.qualified_layers)
            evidence[event.model_id] = evidence.get(event.model_id, 0) | qualified
            self.devices[event.device_id] = replace(device, stored_layers=MappingProxyType(stored),
                                                    qualified_layers=MappingProxyType(evidence))
        elif kind == "DEMAND":
            self._require_model(event)
            self.demand_tokens_per_s[event.model_id] = float(event.details["tokens_per_s"])
            affected, replan = [], False
        if replan:
            for model_id in affected:
                self._decide(event, model_id, kind.lower())
        return self.decisions[before:]

    def _grow_envelopes(self, model_id: str, rows: Sequence[Mapping[str, object]]) -> None:
        """A server's summaries show what each helper sustained: owned layers x measured round trip."""
        owners = self.launched.get(model_id) or self.current.get(model_id) or {}
        for device_id in sorted({owner for owner in owners.values() if owner in self.devices}):
            lowest = [row for row in rows if row["device_id"] == device_id and int(row["calls"]) >= self.policy.sustained_calls]
            if not lowest:
                continue
            row = min(lowest, key=lambda item: int(item["rows"]))
            busy = sum(1 for owner in owners.values() if owner == device_id) * float(row["rpc_ms"])
            if busy > self.envelope_ms.get(device_id, 0.0):
                self.envelope_ms[device_id] = busy

    def _require_device(self, event: PlacementEvent) -> None:
        if event.device_id is None or (event.device_id not in self.devices and event.kind != "DEVICE_JOINED"):
            raise LayerPlacementError(event.kind + " names an unknown device")
        if event.device_id not in self.devices and "device" not in event.details:
            raise LayerPlacementError("DEVICE_JOINED of a new device needs details.device (HelperDevice)")

    def _require_model(self, event: PlacementEvent) -> None:
        if event.model_id not in self.models:
            raise LayerPlacementError(event.kind + " names an unknown model")

    # -- drift -------------------------------------------------------------------------------------
    def _owner_costs(self, model_id: str) -> dict[str, float]:
        """J per layer and token of every owner that could serve this model's first CPU layer now."""
        model = self.models[model_id]
        if not model.cpu_layers:
            return {}
        layer = min(model.cpu_layers)
        problem = self.problem()
        single = replace(problem, models=(replace(model, cpu_layers=(layer,)),), current=MappingProxyType({}))
        owners = [DESKTOP_CPU] + [device.device_id for device in single.devices
                                  if device.serves(model_id) and device.formats.get(model_id)]
        result = {}
        for owner in owners:
            try:
                result[owner] = evaluate_owners(single, {model_id: {layer: owner}}, objective=True)[model_id]
            except LayerPlacementError:
                continue
        return result

    def _drifted(self, model_id: str) -> bool:
        """Did any owner's cost move by more than ``drift_ppm`` since this model was last planned?"""
        planned = self.planned_costs.get(model_id)
        if not planned:
            return True
        now = self._owner_costs(model_id)
        limit = self.policy.drift_ppm / 1e6
        if set(now) != set(planned):
            return True
        return any(planned[owner] <= 0 or abs(value - planned[owner]) / planned[owner] > limit
                   for owner, value in now.items())

    # -- decisions ---------------------------------------------------------------------------------
    def _price(self, model_id: str, owners: Mapping[int, str]) -> float:
        return evaluate_owners(self.problem(), {model_id: owners}, objective=True)[model_id]

    def _record(self, event, model_id, action, before, after, gain, tokens, cost, reason, **details) -> PlacementDecision:
        decision = PlacementDecision(
            event.at_s, event.kind, model_id, action,
            MappingProxyType(owners_to_masks({k: v for k, v in before.items() if v != DESKTOP_CPU})),
            MappingProxyType(owners_to_masks({k: v for k, v in after.items() if v != DESKTOP_CPU})),
            gain, tokens, cost, reason, MappingProxyType(details))
        self.decisions.append(decision)
        return decision

    def _adopt(self, model_id: str, owners: Mapping[int, str], at_s: float, *, runtime: bool) -> None:
        self.target[model_id] = dict(owners)
        if runtime:
            self.current[model_id] = dict(owners)
        self.last_change_s[model_id] = at_s

    def _decide(self, event: PlacementEvent, model_id: str, trigger: str) -> None:
        if model_id not in self.models:
            return
        running = model_id in self.launched
        # 1. mandatory: owners that are gone fall back to the host now (runtime) and at the next launch
        lost = {layer for layer, owner in self.current[model_id].items() if not self._available(owner)}
        if lost:
            before = dict(self.current[model_id])
            self.current[model_id] = {layer: (DESKTOP_CPU if layer in lost else owner) for layer, owner in before.items()}
            self._record(event, model_id, "MASK_OUT_NOW", before, self.current[model_id], 0.0, 0.0, 0.0,
                         "OWNER_UNAVAILABLE", layers=layer_spec(sorted(lost)), running=running)
        lost_target = {layer for layer, owner in self.target[model_id].items() if not self._available(owner)}
        if lost_target:
            self.target[model_id] = {layer: (DESKTOP_CPU if layer in lost_target else owner)
                                     for layer, owner in self.target[model_id].items()}
        mandatory = bool(lost or lost_target)
        # 2. plans: ideal (any shard) and executable (stored + qualified shards only)
        try:
            problem = self.problem()
            ideal = solve_placement(problem)
            executable = solve_placement(restrict_to_executable(problem))
        except LayerPlacementError as error:
            self.errors.append({"at_s": event.at_s, "model_id": model_id, "error": str(error)})
            return
        self.planned_costs[model_id] = self._owner_costs(model_id)
        self.plans.append({"at_s": round(event.at_s, 6), "trigger": event.kind, "model_id": model_id,
                           "ideal": ideal.models[model_id].to_json(),
                           "executable": executable.models[model_id].to_json(),
                           "solver": executable.solver, "gap_bound_j_per_step": executable.gap_bound_j_per_step,
                           "priors_used": [row for row in executable.priors_used if (":" + model_id + ":") in row]})
        candidate = dict(executable.models[model_id].owners)
        tokens = self._tokens(model_id)
        # 3. what the ideal plan would need beyond the executable one (provisioning, qualification)
        needs = [row for row in provisioning_needs(problem, ideal) if row.model_id == model_id]
        if not needs:
            self.recommended.pop(model_id, None)
        signature = tuple(sorted((row.device_id, row.layers, row.needs_shard, row.needs_qualification) for row in needs))
        if needs and self.recommended.get(model_id) != signature:
            self.recommended[model_id] = signature
            gain = executable.models[model_id].energy_j_per_token - ideal.models[model_id].energy_j_per_token
            push_j = sum(row.bytes for row in needs) / self.policy.push_bytes_per_s * self.policy.push_power_w
            needs_json = [row.to_json() for row in needs]
            if any(row.needs_qualification for row in needs):
                self._record(event, model_id, "BLOCKED_QUALIFICATION", candidate, dict(ideal.models[model_id].owners),
                             gain, tokens, push_j, "LAYERS_WITHOUT_PASS_EVIDENCE", needs=needs_json)
            elif self.policy.adopts(gain, tokens, push_j):
                self._record(event, model_id, "PROVISION", candidate, dict(ideal.models[model_id].owners),
                             gain, tokens, push_j, "SHARD_PUSH_PAYS", needs=needs_json)
        # 4. the executable candidate against what runs now (running) or what the next launch uses
        reference = self.current[model_id] if running else self.target[model_id]
        if candidate == reference and candidate == self.target[model_id]:
            return
        gain = self._price(model_id, reference) - self._price(model_id, candidate)
        within_launch = running and all(owner == DESKTOP_CPU or self.launched[model_id].get(layer) == owner
                                        for layer, owner in candidate.items())
        if within_launch:
            # runtime layer-mask control narrows or restores inside the launched ownership: free
            if gain >= self.policy.min_gain_j_per_token or (mandatory and gain >= 0):
                before = dict(self.current[model_id])
                self._adopt(model_id, candidate, event.at_s, runtime=True)
                self._record(event, model_id, "ADOPT_RUNTIME", before, candidate, gain, tokens, 0.0,
                             "WITHIN_LAUNCHED_OWNERSHIP")
            elif gain > 0:
                self._record(event, model_id, "HOLD", reference, candidate, gain, tokens, 0.0, "GAIN_BELOW_MINIMUM")
            return
        since = event.at_s - self.last_change_s.get(model_id, -1e18)
        if since < self.policy.min_dwell_s and not mandatory:
            if gain > 0:
                self._record(event, model_id, "HOLD", reference, candidate, gain, tokens, 0.0, "MIN_DWELL",
                             since_s=round(since, 3))
            return
        if not running:
            if self.policy.adopts(gain, tokens, 0.0) or (mandatory and gain >= 0):
                before = dict(self.target[model_id])
                self._adopt(model_id, candidate, event.at_s, runtime=False)
                self.current[model_id] = dict(candidate)
                self._record(event, model_id, "ADOPT_NEXT_LAUNCH", before, candidate, gain, tokens, 0.0,
                             "NEW_OWNERSHIP_AT_LAUNCH")
            elif gain > 0:
                self._record(event, model_id, "HOLD", reference, candidate, gain, tokens, 0.0, "GAIN_BELOW_MARGIN")
            return
        restart_j = float(self.policy.restart_j.get(model_id, float("inf")))
        if self.policy.adopts(gain, tokens, restart_j):
            before = dict(self.current[model_id])
            self._adopt(model_id, candidate, event.at_s, runtime=True)
            self.launched[model_id] = dict(candidate)
            self._record(event, model_id, "RESTART_NOW", before, candidate, gain, tokens, restart_j,
                         "GAIN_PAYS_RELAUNCH")
        elif self.policy.adopts(gain, tokens, 0.0):
            before = dict(self.target[model_id])
            self._adopt(model_id, candidate, event.at_s, runtime=False)
            self._record(event, model_id, "ADOPT_NEXT_LAUNCH", before, candidate, gain, tokens, 0.0,
                         "RELAUNCH_NOT_PAID_DEFERRED", restart_j=restart_j)
        elif gain > 0:
            self._record(event, model_id, "HOLD", reference, candidate, gain, tokens, restart_j, "GAIN_BELOW_MARGIN")

    # -- views -------------------------------------------------------------------------------------
    def launch_masks(self, model_id: str) -> dict[str, int]:
        """Helper ownership the model's next server launch should use (available devices only)."""
        return {device: mask for device, mask in owners_to_masks(self.target[model_id]).items()
                if device != DESKTOP_CPU and self._available(device)}

    def runtime_layer_mask(self, model_id: str) -> int:
        """Union of the helper layers currently enabled for the running server (runtime FFN control)."""
        return layer_mask(layer for layer, owner in self.current[model_id].items() if owner != DESKTOP_CPU)

    def summary(self) -> dict[str, object]:
        counts: dict[str, int] = {}
        for decision in self.decisions:
            counts[decision.action] = counts.get(decision.action, 0) + 1
        return {
            "schema": LAYER_PLACEMENT_CONTROL_SCHEMA,
            "policy": self.policy.to_json(),
            "decision_counts": dict(sorted(counts.items())),
            "unavailable": {device: sorted(reasons) for device, reasons in sorted(self.unavailable.items()) if reasons},
            "target": {model: {device: layer_spec(mask_layers(mask)) for device, mask in
                               sorted(owners_to_masks(owners).items())} for model, owners in sorted(self.target.items())},
            "errors": list(self.errors),
            "busy_envelope_ms": {device: round(ms, 3) for device, ms in sorted(self.envelope_ms.items())},
            "profile_revision": self.profile.revision,
        }

    def to_json(self) -> dict[str, object]:
        return {**self.summary(), "decisions": [row.to_json() for row in self.decisions], "plans": list(self.plans)}


# --------------------------------------------------------------------------------------------------
# opt-in configuration: dispatch_policy.measured_placement
# --------------------------------------------------------------------------------------------------

MEASURED_PLACEMENT_MODES = ("shadow",)
BUILTIN_RIG_PROFILES = ("4060ti-op15-pixel-v1",)
_CONFIG_FIELDS = frozenset({
    "mode", "rig_profile", "profile_path", "inventory_path", "pixel_transport", "horizon_s", "margin_ppm",
    "min_dwell_s", "drift_ppm", "busy_growth_ppm", "risk",
})


@dataclass(frozen=True)
class MeasuredPlacementConfig:
    """``dispatch_policy.measured_placement`` (opt-in; absent = static layer ownership, byte-identical).

    ``mode`` ``shadow``: the runner replays the finished run's server summaries and membership / thermal
    events through :class:`LayerPlacementController` and writes ``LAYER_PLACEMENT.json`` (plans,
    decisions) and ``LAYER_PLACEMENT_PROFILE.json`` (the profile updated with this run's measurements,
    the input of the next run); execution is unchanged. Costs come from ``profile_path`` (a profile JSON)
    or the builtin ``rig_profile``; models and helpers from ``inventory_path`` or the builtin rig.
    """

    mode: str = "shadow"
    rig_profile: str | None = None
    profile_path: str | None = None
    inventory_path: str | None = None
    pixel_transport: str = "adb-tcp"
    horizon_s: int = 900
    margin_ppm: int = 250_000
    min_dwell_s: int = 120
    drift_ppm: int = 150_000
    busy_growth_ppm: int | None = 250_000
    risk: bool = True

    def __post_init__(self) -> None:
        if self.mode not in MEASURED_PLACEMENT_MODES:
            raise LayerPlacementError("measured_placement mode must be one of " + ", ".join(MEASURED_PLACEMENT_MODES))
        if self.rig_profile is not None and self.rig_profile not in BUILTIN_RIG_PROFILES:
            raise LayerPlacementError("measured_placement rig_profile is unknown")
        if (self.profile_path is None or self.inventory_path is None) and self.rig_profile is None:
            raise LayerPlacementError("measured_placement needs rig_profile or both profile_path and inventory_path")
        for name in ("profile_path", "inventory_path"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value.startswith("/")):
                raise LayerPlacementError("measured_placement " + name + " must be an absolute path")
        if self.pixel_transport not in ("adb-tcp", "aoa-bridge"):
            raise LayerPlacementError("measured_placement pixel_transport must be adb-tcp or aoa-bridge")
        for name in ("horizon_s", "margin_ppm", "min_dwell_s", "drift_ppm"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise LayerPlacementError("measured_placement " + name + " must be an integer >= 0")
        if self.busy_growth_ppm is not None and (type(self.busy_growth_ppm) is not int or self.busy_growth_ppm < 0):
            raise LayerPlacementError("measured_placement busy_growth_ppm must be an integer >= 0 or null")
        if type(self.risk) is not bool:
            raise LayerPlacementError("measured_placement risk must be a boolean")

    @classmethod
    def from_json(cls, value: object) -> "MeasuredPlacementConfig":
        if not isinstance(value, Mapping) or not value:
            raise LayerPlacementError("measured_placement must be a non-empty object")
        unknown = sorted(set(value) - _CONFIG_FIELDS)
        if unknown:
            raise LayerPlacementError("measured_placement has unknown fields: " + ",".join(unknown))
        return cls(**dict(value))

    def to_json(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in sorted(_CONFIG_FIELDS)}

    def policy(self, restart_j: Mapping[str, float] | None = None) -> RebalancePolicy:
        return RebalancePolicy(
            horizon_s=float(self.horizon_s), margin_ppm=self.margin_ppm, min_dwell_s=float(self.min_dwell_s),
            drift_ppm=self.drift_ppm, busy_growth_ppm=self.busy_growth_ppm,
            restart_j=MappingProxyType(dict(restart_j or {})),
            uncertainty_ppm=MappingProxyType({"scaled": 100_000, "prior": 250_000} if self.risk else {}),
            confident_calls=2000 if self.risk else 0,
        )


def measured_placement_from_policy_json(value: Mapping[str, object]) -> tuple[dict[str, object], MeasuredPlacementConfig | None]:
    """Split ``measured_placement`` off a dispatch policy object (the rest is the RuntimeDispatchPolicy)."""
    rest = {key: item for key, item in value.items() if key != "measured_placement"}
    if "measured_placement" not in value:
        return rest, None
    return rest, MeasuredPlacementConfig.from_json(value["measured_placement"])
