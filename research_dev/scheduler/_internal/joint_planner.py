"""Bounded-horizon joint planner over batch formation, model switching, phone shard residency and
helper use, with the sequential rule cascade as incumbent and fallback.

At a decision epoch the planner enumerates joint candidates for the current instant (admit / join /
wait for the running batch; switch now to each queued model; re-provision the primary phone now to
each model with demand, or release it early to prepare a successor; start with or without the helper,
or wait for shards that are being provisioned; late adoption). Candidate 0 is always the sequential
policy's own decision. Each candidate is scored by a depth-limited search: at the next ``depth``
epochs that offer a real choice the search branches again, then the known work is completed by two
base policies (the sequential cascade and the same cascade without joiner serialization) and the
cheaper feasible completion counts.

Objective: energy over a common window from now until the latest completion of the known work
(host, or host plus assumed phone power), plus a terminal value for work left at the horizon. A
candidate that finishes earlier is charged idle power for the rest of the window, so candidates are
compared by ``E - P_idle * t_end`` (the idle power at the end is the same for all of them).
Constraints (``PlannerConstraints``): per-request latency at most 1.25x the desktop-only estimate
(relaxed to the sequential plan when that already exceeds it), start at most
``maximum_displacement_s`` later than in the sequential plan. The planner only sees requests that
have arrived; ``exhaustive_optimum`` is the clairvoyant reference used on small cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import itertools
import math
import time
from typing import Sequence

from .joint_planner_model import CostModel, PlannerConstraints
from .joint_planner_sim import (
    EPSILON_S,
    _describe,
    Action,
    Admit,
    DeferProvision,
    Park,
    Provision,
    SequentialPolicy,
    SetAssist,
    SimOptions,
    SimRequest,
    SimResult,
    SimulationError,
    Simulator,
    Switch,
    Wake,
)


OBJECTIVES = ("fleet", "host")
LATENCY_TOLERANCE_S = 0.5
STALL_OVERRUN_S = 1e7


class JointPlannerError(ValueError):
    pass


@dataclass(frozen=True)
class JointPlannerConfig:
    horizon_s: float = 3600.0
    budget_ms: float = 250.0
    depth: int = 2
    objective: str = "fleet"
    minimum_gain_j: float = 50.0
    constraints: PlannerConstraints = PlannerConstraints()
    max_candidates: int = 16

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise JointPlannerError("objective must be one of " + ", ".join(OBJECTIVES))
        if self.horizon_s <= 0 or self.budget_ms <= 0:
            raise JointPlannerError("horizon_s and budget_ms must be positive")
        if type(self.depth) is not int or self.depth < 0:
            raise JointPlannerError("depth must be a nonnegative integer")
        if self.minimum_gain_j < 0:
            raise JointPlannerError("minimum_gain_j must be nonnegative")

    @classmethod
    def from_json(cls, value: object) -> "JointPlannerConfig":
        if not isinstance(value, dict):
            raise JointPlannerError("joint planner configuration must be an object")
        known = {"horizon_s", "budget_ms", "depth", "objective", "minimum_gain_j"}
        unknown = sorted(set(value) - known - {"mode"})
        if unknown:
            raise JointPlannerError("joint planner configuration has unknown fields: " + ", ".join(unknown))
        return cls(**{k: v for k, v in value.items() if k in known})


@dataclass(frozen=True)
class References:
    """Per-request bounds from the sequential plan and its desktop-only variant."""

    sequential_end_s: dict[str, float]
    sequential_start_s: dict[str, float]
    desktop_end_s: dict[str, float]
    arrival_s: dict[str, float]

    def latency_bound_s(self, request_id: str, constraints: PlannerConstraints) -> float:
        arrival = self.arrival_s[request_id]
        desktop = self.desktop_end_s.get(request_id)
        sequential = self.sequential_end_s.get(request_id)
        bound = math.inf if desktop is None else (
            (desktop - arrival) * constraints.maximum_latency_ppm / 1_000_000
        )
        if sequential is not None and sequential - arrival > bound:
            bound = sequential - arrival
        return bound

    def anchors(self, constraints: PlannerConstraints) -> dict[str, tuple[float, float | None]]:
        """Absolute (deadline, latest start) per request."""
        out = {}
        for request_id, arrival in self.arrival_s.items():
            start = self.sequential_start_s.get(request_id)
            out[request_id] = (
                arrival + self.latency_bound_s(request_id, constraints),
                None if start is None else start + constraints.maximum_displacement_s,
            )
        return out


@dataclass
class Leaf:
    score: float
    energy_j: float
    end_s: float
    feasible: bool
    violations: tuple[str, ...]
    completions: dict[str, float]
    overrun_s: float = 0.0


@dataclass
class PlanResult:
    actions: tuple[Action, ...]
    incumbent: tuple[Action, ...]
    chosen_index: int
    scores: tuple[float | None, ...]
    candidates: tuple[tuple[Action, ...], ...]
    feasible: tuple[bool, ...]
    elapsed_ms: float
    budget_exhausted: bool
    fallback_reason: str | None
    rollouts: int
    predicted_gain_j: float
    # Energy (host, or host plus phones) of each evaluated candidate's leaf (None: not evaluated).
    energies: tuple[float | None, ...] = ()

    @property
    def deviates(self) -> bool:
        return self.chosen_index != 0


@dataclass
class _Search:
    config: JointPlannerConfig
    cost: CostModel
    anchors: dict[str, tuple[float, float | None]]
    deadline: float
    clairvoyant: bool
    sequential: SequentialPolicy
    joint_base: SequentialPolicy
    rollouts: int = 0
    nodes: int = 0
    exhausted: bool = False
    node_limit: int | None = None
    best_score: float = math.inf
    horizon_end_s: float = math.inf
    known: frozenset[str] = field(default_factory=frozenset)
    epoch_reference: dict[str, float] | None = None

    def end_power_w(self) -> float:
        host = self.cost.idle_loaded_w
        if self.config.objective == "host":
            return host
        return host + sum(phone.idle_power_w for phone in self.cost.phones.values())

    def energy(self, sim: Simulator) -> float:
        return sim.host_j + (sim.phone_j if self.config.objective == "fleet" else 0.0)

    def out_of_time(self) -> bool:
        if self.node_limit is not None:
            return self.nodes >= self.node_limit
        return time.perf_counter() >= self.deadline

    # scoring

    def terminal_j(self, sim: Simulator) -> float:
        value = 0.0
        models: set[str] = set()
        for row in sim.server.rows:
            value += row.tokens_left * self.cost.best_token_energy_j(row.request.model)
            models.add(row.request.model)
        for request in sim.queue + (sim.pending if self.clairvoyant else []):
            value += request.output_tokens * self.cost.best_token_energy_j(request.model)
            models.add(request.model)
        for model in models - {sim.server.resident}:
            value += self.cost.model(model).load_energy_j()
        return value

    def leaf(self, sim: Simulator) -> Leaf:
        finished = sim.done() if self.clairvoyant else not (self.known - set(sim.completions))
        energy = self.energy(sim)
        score = energy - self.end_power_w() * sim.now
        violations = []
        overrun = 0.0
        if not finished:
            score += self.terminal_j(sim)
            if sim.now < self.horizon_end_s - EPSILON_S:
                violations.append("stalled")
                overrun += STALL_OVERRUN_S
        admitted = {row.request.request_id: row.admitted_s for row in sim.server.rows}
        for request_id, (deadline_s, latest_start_s) in sorted(self.anchors.items()):
            done = sim.completions.get(request_id)
            end = done.end_s if done is not None else None
            if end is not None and end > deadline_s + LATENCY_TOLERANCE_S:
                violations.append(request_id + ":latency")
                overrun += end - deadline_s
            if end is None and sim.now > deadline_s + LATENCY_TOLERANCE_S:
                violations.append(request_id + ":unfinished")
                overrun += sim.now - deadline_s
            start = done.admitted_s if done is not None else admitted.get(request_id)
            if start is not None and latest_start_s is not None and start > latest_start_s + LATENCY_TOLERANCE_S:
                violations.append(request_id + ":displacement")
                overrun += start - latest_start_s
        if self.epoch_reference is not None:
            slack = self.config.constraints.maximum_epoch_delay_s + LATENCY_TOLERANCE_S
            for request_id, reference_end in sorted(self.epoch_reference.items()):
                done = sim.completions.get(request_id)
                if done is not None and done.end_s > reference_end + slack:
                    violations.append(request_id + ":epoch-delay")
                    overrun += done.end_s - reference_end - slack
        return Leaf(
            score=score, energy_j=energy, end_s=sim.now, feasible=not violations,
            violations=tuple(violations),
            completions={k: v.end_s for k, v in sim.completions.items()},
            overrun_s=overrun,
        )

    def rollout(self, sim: Simulator, policy: SequentialPolicy) -> Leaf:
        self.rollouts += 1
        run = sim.clone(keep_future=self.clairvoyant)
        _run_until(run, policy, self.known, self.clairvoyant, self.horizon_end_s)
        return self.leaf(run)

    def complete(self, sim: Simulator) -> Leaf:
        leaves = [self.rollout(sim, self.sequential), self.rollout(sim, self.joint_base)]
        best = leaves[0]
        for leaf in leaves[1:]:
            if _better(leaf, best):
                best = leaf
        return best

    # search

    def search(self, sim: Simulator, depth: int) -> Leaf:
        """Score ``sim`` at an epoch whose decisions are not yet made."""
        while True:
            if self._finished(sim):
                return self.leaf(sim)
            if depth <= 0 or self.out_of_time():
                if self.out_of_time():
                    self.exhausted = True
                return self.complete(sim)
            candidates = joint_candidates(sim, self.sequential, clairvoyant=self.clairvoyant,
                                          limit=self.config.max_candidates)
            if len(candidates) >= 2:
                return self.branch(sim, candidates, depth)
            apply_epoch(sim, candidates[0])
            if not self._advance(sim):
                return self.leaf(sim)

    def branch(self, sim: Simulator, candidates: Sequence[tuple[Action, ...]], depth: int) -> Leaf:
        best: Leaf | None = None
        for index, candidate in enumerate(candidates):
            if index and self.out_of_time():
                self.exhausted = True
                break
            self.nodes += 1
            child = sim.clone(keep_future=self.clairvoyant)
            apply_epoch(child, candidate)
            if self._advance(child):
                leaf = self.search(child, depth - 1)
            else:
                leaf = self.leaf(child)
            if best is None or _better(leaf, best):
                best = leaf
        assert best is not None
        return best

    def _finished(self, sim: Simulator) -> bool:
        if sim.now >= self.horizon_end_s:
            return True
        if self.clairvoyant:
            return sim.done()
        return not (self.known - set(sim.completions))

    def _advance(self, sim: Simulator) -> bool:
        if self._finished(sim):
            return False
        return _bounded_advance(sim, self.horizon_end_s)


def _better(a: Leaf, b: Leaf) -> bool:
    """Feasible first; among infeasible leaves the smaller total overrun; then energy."""
    if a.feasible != b.feasible:
        return a.feasible
    if not a.feasible and abs(a.overrun_s - b.overrun_s) > LATENCY_TOLERANCE_S:
        return a.overrun_s < b.overrun_s
    return a.score < b.score


def _bounded_advance(sim: Simulator, horizon_end_s: float) -> bool:
    t = sim.next_event_s()
    if not math.isfinite(t):
        return False
    if t >= horizon_end_s:
        sim._advance_to(max(sim.now, horizon_end_s))
        return False
    return sim.advance()


def _run_until(sim: Simulator, policy, known: frozenset[str], clairvoyant: bool, horizon_end_s: float) -> None:
    sim._process_events()
    while True:
        sim.settle(policy)
        if clairvoyant and sim.done() or not clairvoyant and not (known - set(sim.completions)):
            return
        if sim.now >= horizon_end_s or not _bounded_advance(sim, horizon_end_s):
            return


def epoch_decision(sim: Simulator, policy) -> tuple[Action, ...]:
    """The complete decision ``policy`` makes at this instant (fixed point of ``settle``)."""
    probe = sim.clone(keep_future=True)
    probe.applied = []
    probe.settle(policy)
    return tuple(probe.applied)


def apply_epoch(sim: Simulator, actions: Sequence[Action]) -> None:
    for action in actions:
        if isinstance(action, Wake) and (action.at_s <= sim.now or action.at_s in sim.wakes):
            continue
        sim.apply(action)


def _valid(sim: Simulator, actions: Sequence[Action]) -> bool:
    trial = sim.clone(keep_future=False)
    try:
        apply_epoch(trial, actions)
    except SimulationError:
        return False
    return True


def joint_candidates(
    sim: Simulator,
    sequential: SequentialPolicy,
    *,
    clairvoyant: bool = False,
    limit: int = 16,
) -> list[tuple[Action, ...]]:
    """Joint decisions for this instant; index 0 is the sequential cascade's decision."""
    base = epoch_decision(sim, sequential)
    server = sim.server
    cost = sim.cost
    phone_name = cost.primary_phone
    admission: list[tuple[Action, ...]] = []
    if server.loading is None and server.resident is not None:
        model = server.resident
        free = cost.model(model).slots - len(server.rows)
        same = sorted(sim.queued(model), key=lambda r: (r.arrival_s, r.request_id))
        ids = tuple(r.request_id for r in same[:max(free, 0)])
        if ids and server.decoding():
            admission += [(Admit(ids),), (Park(ids),)]
        elif ids and not server.rows:
            if sim.helper_available(model):
                admission += [(Admit(ids, True),), (Admit(ids, False),)]
            else:
                admission.append((Admit(ids, False),))
                ready = sim.helper_ready_at(model)
                if ready is not None:
                    admission.append((Wake(ready),))
        if not server.rows:
            for other in sorted({r.model for r in sim.queue if r.model != model}):
                admission.append((Switch(other),))
    elif server.loading is None and server.resident is None:
        for other in sorted({r.model for r in sim.queue}):
            admission.append((Switch(other),))
    if not admission or not server.rows:
        admission.append(())
    phone: list[tuple[Action, ...]] = [()]
    if sim.options.phones_enabled:
        demand = {r.model for r in sim.queue} | {row.request.model for row in server.rows}
        if clairvoyant:
            demand |= {r.model for r in sim.pending}
        if server.loading:
            demand.add(server.loading)
        targets = sorted(
            m for m in demand
            if cost.model(m).assistable and phone_name in cost.model(m).helper_devices
            and sim.primary.model != m and sim.primary.loading != m
        )
        if sim.phone_free(phone_name) and sim.thermal_ok(phone_name):
            phone += [(Provision(phone_name, m),) for m in targets]
        elif server.assisted and server.rows and sim.primary.loading is None and sim.thermal_ok(phone_name):
            phone += [(SetAssist(False), Provision(phone_name, m)) for m in targets if m != server.resident]
    assist: list[tuple[Action, ...]] = [()]
    if server.decoding() and not server.assisted and sim.helper_available(server.resident):
        assist.append((SetAssist(True),))
    out: list[tuple[Action, ...]] = [base]
    seen = {_key(base)}
    for a, p, s in itertools.product(admission, phone, assist):
        combo = _ordered(a, p, s)
        key = _key(combo)
        if key in seen:
            continue
        seen.add(key)
        if not combo and not math.isfinite(sim.next_event_s()):
            continue
        if _valid(sim, combo):
            out.append(combo)
        if len(out) >= limit:
            break
    return out


def _ordered(admission, phone, assist) -> tuple[Action, ...]:
    releases = tuple(a for a in phone if isinstance(a, SetAssist))
    provisions = tuple(a for a in phone if not isinstance(a, SetAssist))
    return (*releases, *assist, *admission, *provisions)


def _key(actions: Sequence[Action]) -> tuple:
    return tuple(sorted(repr(a) for a in actions if not isinstance(a, DeferProvision)))


def references_for(sim: Simulator, sequential: SequentialPolicy, *, clairvoyant: bool,
                   horizon_end_s: float) -> References:
    known = frozenset(sim.known_ids())
    seq = sim.clone(keep_future=clairvoyant)
    _run_until(seq, sequential, known, clairvoyant, horizon_end_s)
    desk = sim.clone(keep_future=clairvoyant)
    desk.options = replace(desk.options, phones_enabled=False)
    desk.server.assisted = False
    desktop_policy = replace(sequential, helpers=False)
    _run_until(desk, desktop_policy, known, clairvoyant, horizon_end_s)
    ids = set(known) | ({r.request_id for r in sim.pending} if clairvoyant else set())
    arrivals = {}
    for request in sim.queue + [row.request for row in sim.server.rows] + (sim.pending if clairvoyant else []):
        arrivals[request.request_id] = request.arrival_s
    return References(
        sequential_end_s={k: v.end_s for k, v in seq.completions.items() if k in ids},
        sequential_start_s={k: v.admitted_s for k, v in seq.completions.items() if k in ids},
        desktop_end_s={k: v.end_s for k, v in desk.completions.items() if k in ids},
        arrival_s={k: v for k, v in arrivals.items() if k in ids},
    )


class JointPlanner:
    """Policy that plans at every epoch; the sequential cascade is candidate 0 and the fallback."""

    def __init__(
        self,
        config: JointPlannerConfig = JointPlannerConfig(),
        sequential: SequentialPolicy = SequentialPolicy(),
    ) -> None:
        self.config = config
        self.sequential = sequential
        self.joint_base = replace(sequential, serialize_assisted_joiners=False)
        self.plans: list[tuple[float, PlanResult]] = []
        self.anchors: dict[str, tuple[float, float | None]] = {}
        self._decided: tuple[float, int] | None = None

    def decide(self, sim: Simulator) -> list[Action]:
        marker = (sim.now, sim.events)
        if self._decided == marker:
            return []
        self._decided = marker
        plan = self.plan(sim)
        self.plans.append((sim.now, plan))
        return list(plan.actions)

    def plan(self, sim: Simulator, *, clairvoyant: bool = False, node_limit: int | None = None) -> PlanResult:
        started = time.perf_counter()
        deadline = started + self.config.budget_ms / 1e3
        view = sim if clairvoyant else sim.clone(keep_future=False)
        try:
            candidates = joint_candidates(view, self.sequential, clairvoyant=clairvoyant,
                                          limit=self.config.max_candidates)
        except Exception as error:  # noqa: BLE001 - the cascade decision is the fallback
            return self._fallback(sim, started, "CANDIDATES_FAILED: " + type(error).__name__ + ": " + str(error))
        incumbent = candidates[0]
        if len(candidates) == 1:
            return PlanResult(
                actions=incumbent, incumbent=incumbent, chosen_index=0, scores=(None,),
                candidates=tuple(candidates), feasible=(True,), elapsed_ms=_ms(started),
                budget_exhausted=False, fallback_reason=None, rollouts=0, predicted_gain_j=0.0,
            )
        try:
            return self._search_plan(sim, candidates, started, deadline, clairvoyant, node_limit)
        except Exception as error:  # noqa: BLE001
            return self._fallback(sim, started, "SEARCH_FAILED: " + type(error).__name__ + ": " + str(error))

    def _search_plan(self, sim: Simulator, candidates: list[tuple[Action, ...]], started: float,
                     deadline: float, clairvoyant: bool, node_limit: int | None) -> PlanResult:
        incumbent = candidates[0]
        horizon_end = sim.now + self.config.horizon_s
        references = references_for(sim, self.sequential, clairvoyant=clairvoyant, horizon_end_s=horizon_end)
        for request_id, anchor in references.anchors(self.config.constraints).items():
            self.anchors.setdefault(request_id, anchor)
        live = set(references.arrival_s)
        search = _Search(
            config=self.config, cost=sim.cost,
            anchors={k: v for k, v in self.anchors.items() if k in live}, deadline=deadline,
            epoch_reference=dict(references.sequential_end_s),
            clairvoyant=clairvoyant, sequential=self.sequential, joint_base=self.joint_base,
            node_limit=node_limit, horizon_end_s=horizon_end,
            known=frozenset(sim.known_ids()),
        )
        scores: list[float | None] = []
        feasible: list[bool] = []
        leaves: list[Leaf | None] = []
        for index, candidate in enumerate(candidates):
            if index and search.out_of_time():
                search.exhausted = True
                scores.append(None)
                feasible.append(False)
                leaves.append(None)
                continue
            child = sim.clone(keep_future=clairvoyant)
            try:
                apply_epoch(child, candidate)
            except SimulationError:
                scores.append(None)
                feasible.append(False)
                leaves.append(None)
                continue
            if search._advance(child):
                leaf = search.search(child, self.config.depth - 1)
            else:
                leaf = search.leaf(child)
            scores.append(leaf.score)
            feasible.append(leaf.feasible)
            leaves.append(leaf)
        incumbent_leaf = leaves[0]
        if incumbent_leaf is None:
            return self._fallback(sim, started, "INCUMBENT_NOT_EVALUATED")
        chosen = 0
        for index, leaf in enumerate(leaves):
            if index and leaf is not None and _better(leaf, leaves[chosen]):
                chosen = index
        best = leaves[chosen]
        if (chosen and best.feasible == incumbent_leaf.feasible
                and abs(best.overrun_s - incumbent_leaf.overrun_s) <= LATENCY_TOLERANCE_S
                and not best.score < incumbent_leaf.score - self.config.minimum_gain_j):
            chosen = 0
        gain = 0.0 if chosen == 0 else incumbent_leaf.score - leaves[chosen].score
        return PlanResult(
            actions=candidates[chosen], incumbent=incumbent, chosen_index=chosen,
            scores=tuple(scores), candidates=tuple(candidates), feasible=tuple(feasible),
            elapsed_ms=_ms(started), budget_exhausted=search.exhausted, fallback_reason=None,
            rollouts=search.rollouts, predicted_gain_j=gain,
            energies=tuple(None if leaf is None else leaf.energy_j for leaf in leaves),
        )

    def _fallback(self, sim: Simulator, started: float, reason: str) -> PlanResult:
        incumbent = epoch_decision(sim, self.sequential)
        return PlanResult(
            actions=incumbent, incumbent=incumbent, chosen_index=0, scores=(None,),
            candidates=(incumbent,), feasible=(True,), elapsed_ms=_ms(started),
            budget_exhausted=False, fallback_reason=reason, rollouts=0, predicted_gain_j=0.0,
        )


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1e3


@dataclass
class ExhaustiveResult:
    score: float
    leaf: Leaf
    nodes: int
    complete: bool
    decisions: list[tuple[float, str]]
    seeded_by: str


def _state_key(sim: Simulator) -> tuple:
    server = sim.server
    return (
        round(sim.now, 3), tuple(sorted(r.request_id for r in sim.queue)), len(sim.pending),
        server.resident, server.loading, round(server.load_end_s, 3) if server.loading else None,
        tuple((row.request.request_id, row.parked, round(row.tokens_left, 3)) for row in server.rows),
        server.assisted, round(server.prefill_left_s, 3),
        None if server.phase_start_s is None else round(server.phase_start_s, 3), server.phase_rows,
        None if server.last_release_s is None else round(server.last_release_s, 3), server.released_model,
        tuple(
            (p.model, p.loading, round(p.ready_s, 3) if p.loading else None,
             round(max(0.0, p.lease_until_s - sim.now), 3), p.deferred)
            for _, p in sorted(sim.phones.items())
        ),
        tuple(sorted(sim.wakes)), tuple(sorted(sim.bypasses.items())),
    )


def exhaustive_optimum(
    requests: Sequence[SimRequest],
    cost: CostModel,
    options: SimOptions = SimOptions(),
    *,
    config: JointPlannerConfig = JointPlannerConfig(),
    sequential: SequentialPolicy = SequentialPolicy(),
    node_limit: int = 200_000,
) -> ExhaustiveResult:
    """Clairvoyant branch-and-bound optimum over the joint candidate space (small cases only).

    Future arrivals are visible and waiting for the next event is a candidate at every epoch.
    The incumbent is seeded with the sequential cascade, its no-serialization variant and the online
    planner; states reached again at no lower cost are pruned; the lower bound adds, per remaining
    request, its tokens at the cheapest marginal energy per token, its prefill, and one load per
    non-resident model with remaining work.
    """
    root = Simulator(requests, cost, options)
    root.record_decisions = False
    root._process_events()
    horizon_end = config.horizon_s
    references = references_for(root, sequential, clairvoyant=True, horizon_end_s=horizon_end)
    joint_base = replace(sequential, serialize_assisted_joiners=False)
    search = _Search(
        config=config, cost=cost, anchors=references.anchors(config.constraints), deadline=math.inf,
        clairvoyant=True,
        sequential=sequential, joint_base=joint_base,
        node_limit=None, horizon_end_s=horizon_end, known=frozenset(r.request_id for r in requests),
    )
    end_power = search.end_power_w()
    marginal = {
        name: _marginal_token_energy(cost, name, end_power, config.objective) for name in cost.models
    }
    recorded = dict(options.load_s_by_model)
    load_marginal = {}
    for name in cost.models:
        costs = cost.model(name)
        longest = max((costs.load_s, *recorded.get(name, ())))
        load_marginal[name] = costs.load_fixed_j + min(
            (costs.load_power_w - cost.idle_loaded_w) * d for d in (costs.load_s, longest)
        )
    prefill_power = {name: cost.model(name).power_w(1, False) - cost.idle_loaded_w for name in cost.models}
    state: dict = {"nodes": 0, "best": None, "complete": True, "memo": {}, "seed": ""}

    def offer(leaf: Leaf, trail, label: str) -> None:
        best = state["best"]
        if best is None or _better(leaf, best[0]):
            state["best"] = (leaf, list(trail))
            state["seed"] = label

    for label, policy in (
        ("sequential", sequential), ("joint-base", joint_base),
        ("planner", JointPlanner(replace(config, budget_ms=max(config.budget_ms, 1000.0)), sequential)),
    ):
        seeded = Simulator(requests, cost, options)
        seeded.run(policy)
        offer(search.leaf(seeded), seeded.decisions, label)

    def lower_bound(sim: Simulator) -> float:
        value = search.energy(sim) - end_power * sim.now
        models = set()
        for row in sim.server.rows:
            value += row.tokens_left * marginal[row.request.model]
            models.add(row.request.model)
        for request in sim.queue + sim.pending:
            costs = cost.model(request.model)
            value += request.output_tokens * marginal[request.model]
            value += prefill_power[request.model] * costs.prefill_s(request.input_tokens)
            models.add(request.model)
        for name in models - {sim.server.resident, sim.server.loading}:
            value += load_marginal[name]
        return value

    def dominated(sim: Simulator) -> bool:
        key = _state_key(sim)
        value = search.energy(sim) - end_power * sim.now
        seen = state["memo"].get(key)
        if seen is not None and seen <= value + 1e-6:
            return True
        state["memo"][key] = value
        return False

    def dfs(sim: Simulator, trail: list[tuple[float, str]]) -> None:
        while True:
            if sim.done() or sim.now >= horizon_end:
                offer(search.leaf(sim), trail, "search")
                return
            best = state["best"]
            if best[0].feasible and lower_bound(sim) >= best[0].score - 1e-6:
                return
            if dominated(sim):
                return
            if state["nodes"] >= node_limit:
                state["complete"] = False
                return
            candidates = joint_candidates(sim, sequential, clairvoyant=True, limit=10_000)
            if len(candidates) == 1:
                apply_epoch(sim, candidates[0])
                trail = trail + [(round(sim.now, 3), _describe_all(candidates[0]))] if candidates[0] else trail
                if not _bounded_advance(sim, horizon_end):
                    offer(search.leaf(sim), trail, "search")
                    return
                continue
            ordered = sorted(candidates, key=lambda c: 0 if any(isinstance(a, (Admit, Switch)) for a in c) else 1)
            for candidate in ordered:
                state["nodes"] += 1
                child = sim.clone(keep_future=True)
                apply_epoch(child, candidate)
                step = trail + ([(round(sim.now, 3), _describe_all(candidate))] if candidate else [])
                if _bounded_advance(child, horizon_end):
                    dfs(child, step)
                else:
                    offer(search.leaf(child), step, "search")
            return

    dfs(root.clone(keep_future=True), [])
    leaf, trail = state["best"]
    return ExhaustiveResult(score=leaf.score, leaf=leaf, nodes=state["nodes"],
                            complete=state["complete"], decisions=trail, seeded_by=state["seed"])


@dataclass(frozen=True)
class Phase:
    """One server phase: a same-model request set served without a reload, with or without helpers."""

    model: str
    request_ids: tuple[str, ...]
    assisted: bool


@dataclass(frozen=True)
class SchedulePolicy:
    """Executes a clairvoyant phase schedule.

    Phases run in order; a phase starts when the previous one has finished (the server switches model
    at once and then waits for the phase's members, which is how a hold is expressed); members that
    arrive while their phase decodes join the running batch. The primary phone is provisioned as
    early as it is free to the model of the current assisted phase, else of the next assisted phase
    (preparing the successor while a host-only phase runs); a phase that wanted the helper adopts it
    late when the shards become ready.
    """

    phases: tuple[Phase, ...]

    def decide(self, sim: Simulator) -> list[Action]:
        index = next(
            (i for i, phase in enumerate(self.phases)
             if any(rid not in sim.completions for rid in phase.request_ids)),
            None,
        )
        if index is None:
            return []
        phase = self.phases[index]
        server = sim.server
        actions: list[Action] = []
        members = [r for r in sim.queue if r.request_id in phase.request_ids]
        members.sort(key=lambda r: (r.arrival_s, r.request_id))
        if server.loading is None:
            if server.resident != phase.model:
                if not server.rows:
                    actions.append(Switch(phase.model))
            else:
                free = sim.cost.model(phase.model).slots - len(server.rows)
                ids = tuple(r.request_id for r in members[:max(free, 0)])
                if ids:
                    use = phase.assisted and sim.helper_available(phase.model)
                    actions.append(Admit(ids, use if not server.rows else None))
                elif (phase.assisted and server.decoding() and not server.assisted
                      and sim.helper_available(phase.model)
                      and max(r.tokens_left for r in server.decoding()) >= sim.cost.late_adoption_minimum_tokens):
                    actions.append(SetAssist(True))
        if sim.options.phones_enabled:
            actions.extend(self._provision(sim, index, actions))
        return actions

    def _provision(self, sim: Simulator, index: int, pending: Sequence[Action]) -> list[Action]:
        cost = sim.cost
        name = cost.primary_phone
        target = next(
            (p.model for p in self.phases[index:] if p.assisted and cost.model(p.model).assistable), None,
        )
        phone = sim.phones[name]
        if target is None or phone.model == target or phone.loading == target:
            return []
        if any(isinstance(a, Admit) and a.assisted for a in pending):
            return []
        if not sim.phone_free(name) or not sim.thermal_ok(name):
            return []
        return [Provision(name, target)]


def ordered_phase_schedules(requests: Sequence[SimRequest], cost: CostModel,
                            *, phones: bool = True) -> list[tuple[Phase, ...]]:
    """All orders x same-model batchings x helper choices for a small request set."""
    by_model: dict[str, list[str]] = {}
    for request in sorted(requests, key=lambda r: (r.arrival_s, r.request_id)):
        by_model.setdefault(request.model, []).append(request.request_id)
    per_model = {m: list(_set_partitions(ids)) for m, ids in by_model.items()}
    schedules: set[tuple[Phase, ...]] = set()
    for combo in itertools.product(*per_model.values()):
        blocks = [(model, tuple(block)) for model, parts in zip(per_model, combo) for block in parts]
        for order in itertools.permutations(blocks):
            flags = [phones and cost.model(m).assistable for m, _ in order]
            for bits in itertools.product(*[(True, False) if f else (False,) for f in flags]):
                schedules.add(tuple(Phase(m, ids, bit) for (m, ids), bit in zip(order, bits)))
    return sorted(schedules, key=repr)


def _set_partitions(items: Sequence[str]):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for partition in _set_partitions(rest):
        yield [[first], *partition]
        for i in range(len(partition)):
            yield [*partition[:i], [first, *partition[i]], *partition[i + 1:]]


@dataclass
class ScheduleOptimum:
    score: float
    leaf: Leaf
    schedule: tuple[Phase, ...] | None
    evaluated: int
    source: str
    decisions: list[tuple[float, str]]


def schedule_space_optimum(
    requests: Sequence[SimRequest],
    cost: CostModel,
    options: SimOptions = SimOptions(),
    *,
    config: JointPlannerConfig = JointPlannerConfig(),
    sequential: SequentialPolicy = SequentialPolicy(),
    max_schedules: int = 200_000,
) -> ScheduleOptimum:
    """Exact minimum over every phase schedule (clairvoyant), seeded with the sequential and planner runs.

    The same feasibility rules as the planner apply (latency vs the desktop-only estimate from time 0,
    start displacement vs the sequential plan).
    """
    root = Simulator(requests, cost, options)
    root._process_events()
    references = references_for(root, sequential, clairvoyant=True, horizon_end_s=config.horizon_s)
    search = _Search(
        config=config, cost=cost, anchors=references.anchors(config.constraints), deadline=math.inf,
        clairvoyant=True,
        sequential=sequential, joint_base=replace(sequential, serialize_assisted_joiners=False),
        horizon_end_s=config.horizon_s, known=frozenset(r.request_id for r in requests),
    )
    best: tuple[Leaf, tuple[Phase, ...] | None, str, list] | None = None
    seeds = (
        ("sequential", sequential),
        ("planner", JointPlanner(replace(config, budget_ms=max(config.budget_ms, 1000.0)), sequential)),
    )
    for label, policy in seeds:
        sim = Simulator(requests, cost, options)
        sim.run(policy)
        leaf = search.leaf(sim)
        if best is None or _better(leaf, best[0]):
            best = (leaf, None, label, sim.decisions)
    schedules = ordered_phase_schedules(requests, cost, phones=options.phones_enabled)
    if len(schedules) > max_schedules:
        raise JointPlannerError("schedule space too large: %d" % len(schedules))
    for schedule in schedules:
        sim = Simulator(requests, cost, replace(options, max_events=20_000))
        try:
            sim.run(SchedulePolicy(schedule))
        except SimulationError:
            continue
        if not sim.done():
            continue
        leaf = search.leaf(sim)
        if _better(leaf, best[0]):
            best = (leaf, schedule, "schedule", sim.decisions)
    assert best is not None
    leaf, schedule, source, decisions = best
    return ScheduleOptimum(score=leaf.score, leaf=leaf, schedule=schedule, evaluated=len(schedules),
                           source=source, decisions=list(decisions))


def clairvoyant_leaf(
    requests: Sequence[SimRequest],
    cost: CostModel,
    options: SimOptions,
    finished: Simulator,
    *,
    config: JointPlannerConfig = JointPlannerConfig(),
    sequential: SequentialPolicy = SequentialPolicy(),
) -> Leaf:
    """Score and ex-post constraint check of a finished run against references computed at time 0."""
    root = Simulator(requests, cost, options)
    root._process_events()
    references = references_for(root, sequential, clairvoyant=True, horizon_end_s=config.horizon_s)
    search = _Search(
        config=config, cost=cost, anchors=references.anchors(config.constraints), deadline=math.inf,
        clairvoyant=True,
        sequential=sequential, joint_base=sequential, horizon_end_s=config.horizon_s,
        known=frozenset(r.request_id for r in requests),
    )
    return search.leaf(finished)


def _describe_all(actions: Sequence[Action]) -> str:
    return "; ".join(_describe(a) for a in actions)


def _marginal_token_energy(cost: CostModel, name: str, end_power_w: float, objective: str) -> float:
    costs = cost.model(name)
    best = math.inf
    phones_idle = sum(p.idle_power_w for p in cost.phones.values()) if objective == "fleet" else 0.0
    for b in range(1, costs.slots + 1):
        host = costs.power_w(b, False) + phones_idle
        best = min(best, (host - end_power_w) * costs.step_s(b, False) / b)
        if costs.assistable:
            phones = sum(
                (cost.phones[d].active_power_w if d in costs.helper_devices else cost.phones[d].idle_power_w)
                for d in cost.phones
            ) if objective == "fleet" else 0.0
            best = min(best, (costs.power_w(b, True) + phones - end_power_w) * costs.step_s(b, True) / b)
    return max(0.0, best)


def evaluate_policy(requests: Sequence[SimRequest], policy, cost: CostModel,
                    options: SimOptions = SimOptions()) -> SimResult:
    return Simulator(requests, cost, options).run(policy)
