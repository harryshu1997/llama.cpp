"""Discrete-event simulator of the desktop server + phone helpers, and the sequential policy emulation.

The simulator models what the dispatcher's decisions change: one exclusive desktop model residency
(loads evict), server slots per model, prefill, decode steps whose period and host power depend on the
batch size and on phone assistance, phone shard re-provisioning (the primary phone holds one model's
shards; re-provisioning takes ``sessions x session_load_s``), capacity-1 phone lanes held by an assisted
batch (plus the lag after a multi-row cohort ends), and thermal exclusion intervals. Energy is
integrated exactly between events (piecewise-constant power).

Policies are stateless: ``decide(sim)`` reads the simulator (queue, server, phones, per-request bypass
counts, arrival history) and returns actions for the current instant. ``SequentialPolicy`` emulates the
rule cascade of ``paper_config_v1`` (2.5 s cohort window, work-conserving admission, model affinity,
continuous join as desktop parent only while no phone lane is held, residency hysteresis, early phone
re-provisioning that follows the desktop commitment, late helper adoption). ``LegacyPolicy`` is the
all-desktop arrival-order baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Iterable, Protocol, Sequence

from .joint_planner_model import CostModel


EPSILON_S = 1e-9


class SimulationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SimRequest:
    request_id: str
    model: str
    arrival_s: float
    input_tokens: int
    output_tokens: int


# Actions ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Admit:
    """Admit queued requests of the resident model: start a batch, or join the running one.

    ``assisted`` applies when a new batch starts (None: use the helper when it is available).
    A joiner of an assisted batch shares its phone assistance.
    """

    request_ids: tuple[str, ...]
    assisted: bool | None = None


@dataclass(frozen=True)
class Park:
    """Admit behind the running assisted batch: prefill now, decode when the batch ends."""

    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class Switch:
    model: str
    owner: str | None = None


@dataclass(frozen=True)
class Provision:
    phone: str
    model: str


@dataclass(frozen=True)
class SetAssist:
    on: bool


@dataclass(frozen=True)
class Wake:
    at_s: float


@dataclass(frozen=True)
class DeferProvision:
    """Record a re-provision target that could not start because the phone was in use."""

    phone: str
    model: str | None


Action = Admit | Park | Switch | Provision | SetAssist | Wake | DeferProvision


class Policy(Protocol):
    def decide(self, sim: "Simulator") -> list[Action]:
        ...


# State -----------------------------------------------------------------------------------------


@dataclass
class Row:
    request: SimRequest
    admitted_s: float
    tokens_left: float
    parked: bool = False
    decode_start_s: float | None = None
    first_token_s: float | None = None
    assisted_tokens: float = 0.0


@dataclass
class ServerState:
    resident: str | None = None
    loading: str | None = None
    load_end_s: float = 0.0
    rows: list[Row] = field(default_factory=list)
    assisted: bool = False
    phase_start_s: float | None = None
    phase_rows: int = 0
    prefill_left_s: float = 0.0
    last_release_s: float | None = None
    released_model: str | None = None
    load_owner: str | None = None
    loads: int = 0

    @property
    def model(self) -> str | None:
        return self.loading or self.resident

    def decoding(self) -> list[Row]:
        return [row for row in self.rows if not row.parked]

    def busy(self) -> bool:
        return bool(self.rows) or self.loading is not None


@dataclass
class PhoneState:
    name: str
    model: str | None = None
    loading: str | None = None
    ready_s: float = 0.0
    lease_until_s: float = 0.0
    deferred: str | None = None
    provisions: int = 0


@dataclass(frozen=True)
class Completion:
    request_id: str
    model: str
    arrival_s: float
    admitted_s: float
    decode_start_s: float
    first_token_s: float
    end_s: float
    output_tokens: int
    assisted_tokens: float

    @property
    def latency_s(self) -> float:
        return self.end_s - self.arrival_s


@dataclass(frozen=True)
class SimOptions:
    phones_enabled: bool = True
    cohort_lease_release_lag_s: float = 1.0
    thermal_exclusions: tuple[tuple[str, float, float], ...] = ()
    load_s_by_model: tuple[tuple[str, tuple[float, ...]], ...] = ()
    horizon_s: float | None = None
    max_events: int = 200_000


@dataclass
class SimResult:
    completions: dict[str, Completion]
    end_s: float
    host_j: float
    phone_j: float
    state_s: dict[str, float]
    loads: int
    provisions: int
    unfinished: tuple[str, ...]
    decisions: list[tuple[float, str]]

    @property
    def fleet_j(self) -> float:
        return self.host_j + self.phone_j


# Simulator -------------------------------------------------------------------------------------


class Simulator:
    def __init__(
        self,
        requests: Iterable[SimRequest],
        cost: CostModel,
        options: SimOptions = SimOptions(),
        *,
        start_s: float = 0.0,
    ) -> None:
        self.cost = cost
        self.options = options
        self.now = start_s
        self.pending = sorted(requests, key=lambda r: (r.arrival_s, r.request_id))
        ids = [r.request_id for r in self.pending]
        if len(ids) != len(set(ids)):
            raise SimulationError("request ids must be unique")
        for request in self.pending:
            cost.model(request.model)
        self.queue: list[SimRequest] = []
        self.bypasses: dict[str, int] = {}
        self.arrival_gaps: dict[str, list[float]] = {}
        self.last_arrival: dict[str, float] = {}
        self.server = ServerState()
        self.phones = {
            name: PhoneState(name=name, model=phone.fixed_model)
            for name, phone in cost.phones.items()
        }
        self.completions: dict[str, Completion] = {}
        self.wakes: set[float] = set()
        self.host_j = 0.0
        self.phone_j = 0.0
        self.state_s: dict[str, float] = {}
        self.decisions: list[tuple[float, str]] = []
        self.events = 0
        self._load_index: dict[str, int] = {}
        self.record_decisions = True
        self.applied: list[Action] | None = None

    # -- copying -------------------------------------------------------------------------------

    def clone(self, *, keep_future: bool = True) -> "Simulator":
        other = Simulator.__new__(Simulator)
        other.cost = self.cost
        other.options = self.options
        other.now = self.now
        other.pending = list(self.pending) if keep_future else []
        other.queue = list(self.queue)
        other.bypasses = dict(self.bypasses)
        other.arrival_gaps = {k: list(v) for k, v in self.arrival_gaps.items()}
        other.last_arrival = dict(self.last_arrival)
        other.server = replace(self.server, rows=[replace(row) for row in self.server.rows])
        other.phones = {k: replace(v) for k, v in self.phones.items()}
        other.completions = dict(self.completions)
        other.wakes = set(self.wakes)
        other.host_j = self.host_j
        other.phone_j = self.phone_j
        other.state_s = dict(self.state_s)
        other.decisions = []
        other.events = self.events
        other._load_index = dict(self._load_index)
        other.record_decisions = False
        other.applied = None
        return other

    def without_phones(self) -> "Simulator":
        other = self.clone(keep_future=False)
        other.options = replace(self.options, phones_enabled=False)
        other.server.assisted = False
        return other

    # -- views ---------------------------------------------------------------------------------

    @property
    def primary(self) -> PhoneState:
        return self.phones[self.cost.primary_phone]

    def thermal_ok(self, phone: str, at_s: float | None = None) -> bool:
        t = self.now if at_s is None else at_s
        return not any(
            name == phone and start <= t < end for name, start, end in self.options.thermal_exclusions
        )

    def helper_available(self, model: str | None) -> bool:
        if model is None or not self.options.phones_enabled:
            return False
        costs = self.cost.model(model)
        if not costs.assistable:
            return False
        for device in costs.helper_devices:
            phone = self.phones[device]
            if phone.loading is not None or phone.model != model or not self.thermal_ok(device):
                return False
        return True

    def helper_ready_at(self, model: str) -> float | None:
        """When the primary phone finishes provisioning ``model`` (None: not provisioning it)."""
        phone = self.primary
        return phone.ready_s if phone.loading == model else None

    def phone_leased(self, name: str) -> bool:
        phone = self.phones[name]
        if self.now < phone.lease_until_s - EPSILON_S:
            return True
        server = self.server
        return bool(
            server.rows and server.assisted and server.resident is not None
            and name in self.cost.model(server.resident).helper_devices
        )

    def phone_free(self, name: str) -> bool:
        return not self.phone_leased(name) and self.phones[name].loading is None

    def queued(self, model: str | None = None) -> list[SimRequest]:
        return [r for r in self.queue if model is None or r.model == model]

    def queued_other_than(self, model: str | None) -> list[SimRequest]:
        return [r for r in self.queue if r.model != model]

    def just_arrived(self) -> bool:
        return any(abs(r.arrival_s - self.now) <= EPSILON_S for r in self.queue)

    def remaining_tokens(self, model: str) -> float:
        active = sum(row.tokens_left for row in self.server.rows if row.request.model == model)
        return active + sum(r.output_tokens for r in self.queue if r.model == model)

    def done(self) -> bool:
        return not self.pending and not self.queue and not self.server.rows

    def known_ids(self) -> set[str]:
        return {r.request_id for r in self.queue} | {row.request.request_id for row in self.server.rows}

    def arrival_probability(self, model: str, window_s: float) -> float | None:
        """P(same-model arrival within window) from the learned mean gap (hysteresis estimator)."""
        gaps = self.arrival_gaps.get(model, [])
        if len(gaps) < 2:
            return None
        mean = gaps[0]
        for gap in gaps[1:]:
            mean = 0.75 * mean + 0.25 * gap
        return 1.0 - math.exp(-window_s / max(mean, EPSILON_S))

    # -- actions -------------------------------------------------------------------------------

    def apply(self, action: Action) -> None:
        if isinstance(action, Admit):
            self._admit(action.request_ids, action.assisted, parked=False)
        elif isinstance(action, Park):
            self._admit(action.request_ids, None, parked=True)
        elif isinstance(action, Switch):
            self._switch(action.model)
            self.server.load_owner = action.owner
        elif isinstance(action, Provision):
            self._provision(action.phone, action.model)
        elif isinstance(action, SetAssist):
            self._set_assist(action.on)
        elif isinstance(action, Wake):
            if action.at_s > self.now + EPSILON_S:
                self.wakes.add(action.at_s)
        elif isinstance(action, DeferProvision):
            self.phones[action.phone].deferred = action.model
        else:
            raise SimulationError("unknown action " + repr(action))
        if self.applied is not None:
            self.applied.append(action)
        if self.record_decisions and not isinstance(action, Wake):
            self.decisions.append((round(self.now, 3), _describe(action)))

    def _take(self, request_ids: Sequence[str]) -> list[SimRequest]:
        by_id = {r.request_id: r for r in self.queue}
        missing = [rid for rid in request_ids if rid not in by_id]
        if missing or not request_ids:
            raise SimulationError("admit names requests that are not queued: " + repr(missing))
        taken = [by_id[rid] for rid in request_ids]
        self.queue = [r for r in self.queue if r.request_id not in set(request_ids)]
        return taken

    def _admit(self, request_ids: Sequence[str], assisted: bool | None, *, parked: bool) -> None:
        server = self.server
        if server.loading is not None or server.resident is None:
            raise SimulationError("admission needs a resident model")
        requests = self._take(request_ids)
        if any(r.model != server.resident for r in requests):
            raise SimulationError("admission of another model than the resident one")
        costs = self.cost.model(server.resident)
        if len(server.rows) + len(requests) > costs.slots:
            raise SimulationError("admission exceeds the server slots")
        if parked and not server.decoding():
            raise SimulationError("parking needs a running batch")
        new_batch = not server.rows
        for request in sorted(requests, key=lambda r: (r.arrival_s, r.request_id)):
            older = [q for q in self.queue if q.model != request.model and q.arrival_s < request.arrival_s]
            for other in older:
                self.bypasses[other.request_id] = self.bypasses.get(other.request_id, 0) + 1
            server.rows.append(Row(request=request, admitted_s=self.now,
                                   tokens_left=float(request.output_tokens), parked=parked))
            server.prefill_left_s += costs.prefill_s(request.input_tokens)
        if new_batch:
            use = self.helper_available(server.resident) if assisted is None else assisted
            if use and not self.helper_available(server.resident):
                raise SimulationError("assisted batch without an available helper")
            server.assisted = use
            server.phase_start_s = self.now
            server.phase_rows = len(server.rows)
        else:
            server.phase_rows = max(server.phase_rows, len(server.rows))

    def _switch(self, model: str) -> None:
        server = self.server
        if server.rows or server.loading is not None:
            raise SimulationError("switch while the server is busy")
        if server.resident == model:
            raise SimulationError("switch to the resident model")
        costs = self.cost.model(model)
        duration = costs.load_s
        recorded = dict(self.options.load_s_by_model).get(model, ())
        index = self._load_index.get(model, 0)
        if index < len(recorded):
            duration = recorded[index]
        self._load_index[model] = index + 1
        server.resident = None
        server.loading = model
        server.load_end_s = self.now + duration
        server.assisted = False
        server.loads += 1
        self.host_j += costs.load_fixed_j

    def _provision(self, name: str, model: str) -> None:
        phone = self.phones[name]
        spec = self.cost.phones[name]
        if not self.options.phones_enabled:
            raise SimulationError("provisioning with phones disabled")
        if spec.fixed_model is not None:
            raise SimulationError("phone " + name + " has fixed shards")
        if not self.phone_free(name):
            raise SimulationError("provisioning a phone that is in use")
        if not self.thermal_ok(name):
            raise SimulationError("provisioning a thermally excluded phone")
        if not self.cost.model(model).assistable or name not in self.cost.model(model).helper_devices:
            raise SimulationError("model " + model + " has no shards for " + name)
        phone.model = None
        phone.loading = model
        phone.ready_s = self.now + spec.provision_s
        phone.deferred = None
        phone.provisions += 1

    def _set_assist(self, on: bool) -> None:
        server = self.server
        if not server.rows:
            raise SimulationError("assistance toggle without a batch")
        if on and not self.helper_available(server.resident):
            raise SimulationError("assistance without an available helper")
        if not on and server.assisted:
            self._release_lease(server)
        server.assisted = on

    def _release_lease(self, server: ServerState) -> None:
        lag = self.options.cohort_lease_release_lag_s if server.phase_rows > 1 else 0.0
        for device in self.cost.model(server.resident).helper_devices:
            self.phones[device].lease_until_s = max(self.phones[device].lease_until_s, self.now + lag)

    # -- time ----------------------------------------------------------------------------------

    def _step_s(self) -> float:
        server = self.server
        return self.cost.model(server.resident).step_s(len(server.decoding()), server.assisted)

    def next_event_s(self) -> float:
        times = []
        if self.pending:
            times.append(self.pending[0].arrival_s)
        server = self.server
        if server.loading is not None:
            times.append(server.load_end_s)
        elif server.rows:
            if server.prefill_left_s > EPSILON_S:
                times.append(self.now + server.prefill_left_s)
            elif server.decoding():
                times.append(self.now + min(r.tokens_left for r in server.decoding()) * self._step_s())
        for phone in self.phones.values():
            if phone.loading is not None:
                times.append(phone.ready_s)
            if phone.lease_until_s > self.now + EPSILON_S:
                times.append(phone.lease_until_s)
        for _, start, end in self.options.thermal_exclusions:
            for t in (start, end):
                if t > self.now + EPSILON_S:
                    times.append(t)
        times.extend(t for t in self.wakes if t > self.now + EPSILON_S)
        if self.options.horizon_s is not None:
            times.append(self.options.horizon_s)
        return min(times) if times else math.inf

    def _power(self) -> tuple[float, float, str]:
        server = self.server
        cost = self.cost
        if server.loading is not None:
            host, state = cost.model(server.loading).load_power_w, "load"
        elif server.rows and server.prefill_left_s > EPSILON_S:
            host, state = cost.model(server.resident).power_w(1, False), "prefill"
        elif server.decoding():
            b = len(server.decoding())
            host = cost.model(server.resident).power_w(b, server.assisted)
            state = ("assisted" if server.assisted else "desktop") + "-b" + str(b)
        elif server.resident is not None:
            host, state = cost.idle_loaded_w, "idle-loaded"
        else:
            host, state = cost.idle_unloaded_w, "idle-unloaded"
        phone = 0.0
        decoding = server.loading is None and server.decoding() and server.prefill_left_s <= EPSILON_S
        helpers = (
            cost.model(server.resident).helper_devices
            if server.resident is not None and decoding and server.assisted else ()
        )
        for name, spec in cost.phones.items():
            active = self.phones[name].loading is not None or name in helpers
            phone += spec.active_power_w if active else spec.idle_power_w
        return host, phone, state

    def _advance_to(self, t: float) -> None:
        dt = t - self.now
        if dt < -EPSILON_S:
            raise SimulationError("time went backwards")
        if dt <= 0:
            return
        host, phone, state = self._power()
        self.host_j += host * dt
        self.phone_j += phone * dt
        self.state_s[state] = self.state_s.get(state, 0.0) + dt
        server = self.server
        if server.loading is None and server.rows:
            if server.prefill_left_s > EPSILON_S:
                server.prefill_left_s = max(0.0, server.prefill_left_s - dt)
            elif server.decoding():
                step = self._step_s()
                tokens = dt / step
                for row in server.decoding():
                    if row.decode_start_s is None:
                        row.decode_start_s = self.now
                        row.first_token_s = self.now + step
                    row.tokens_left -= tokens
                    if server.assisted:
                        row.assisted_tokens += tokens
        self.now = t

    def _process_events(self) -> None:
        while self.pending and self.pending[0].arrival_s <= self.now + EPSILON_S:
            request = self.pending.pop(0)
            last = self.last_arrival.get(request.model)
            if last is not None:
                self.arrival_gaps.setdefault(request.model, []).append(request.arrival_s - last)
            self.last_arrival[request.model] = request.arrival_s
            self.queue.append(request)
        server = self.server
        if server.loading is not None and server.load_end_s <= self.now + EPSILON_S:
            server.resident = server.loading
            server.loading = None
        for phone in self.phones.values():
            if phone.loading is not None and phone.ready_s <= self.now + EPSILON_S:
                phone.model = phone.loading
                phone.loading = None
        finished = [row for row in server.decoding() if row.tokens_left <= 1e-6]
        for row in finished:
            server.rows.remove(row)
            self.completions[row.request.request_id] = Completion(
                request_id=row.request.request_id, model=row.request.model,
                arrival_s=row.request.arrival_s, admitted_s=row.admitted_s,
                decode_start_s=row.decode_start_s if row.decode_start_s is not None else self.now,
                first_token_s=row.first_token_s if row.first_token_s is not None else self.now,
                end_s=self.now, output_tokens=row.request.output_tokens,
                assisted_tokens=row.assisted_tokens,
            )
        if finished and not server.decoding():
            if server.rows:
                if server.assisted:
                    self._release_lease(server)
                for row in server.rows:
                    row.parked = False
                server.phase_start_s = self.now
                server.phase_rows = len(server.rows)
                server.assisted = self.helper_available(server.resident)
            else:
                if server.assisted:
                    self._release_lease(server)
                server.assisted = False
                server.last_release_s = self.now
                server.released_model = server.resident
                server.phase_start_s = None
                server.phase_rows = 0
        self.wakes = {t for t in self.wakes if t > self.now + EPSILON_S}

    def settle(self, policy: Policy, *, rounds: int = 8) -> None:
        for _ in range(rounds):
            actions = policy.decide(self)
            progress = False
            for action in actions:
                if isinstance(action, Wake) and (action.at_s in self.wakes or action.at_s <= self.now):
                    continue
                self.apply(action)
                progress = progress or not isinstance(action, Wake)
            if not progress:
                return

    def advance(self) -> bool:
        t = self.next_event_s()
        if not math.isfinite(t):
            return False
        if self.options.horizon_s is not None and t >= self.options.horizon_s - EPSILON_S:
            self._advance_to(max(self.now, self.options.horizon_s))
            return False
        self._advance_to(t)
        self._process_events()
        self.events += 1
        if self.events > self.options.max_events:
            raise SimulationError("event limit exceeded")
        return True

    def run(self, policy: Policy) -> SimResult:
        self._process_events()
        while True:
            self.settle(policy)
            if self.done():
                break
            if not self.advance():
                break
        return self.result()

    def result(self) -> SimResult:
        unfinished = tuple(sorted(
            [r.request_id for r in self.pending] + [r.request_id for r in self.queue]
            + [row.request.request_id for row in self.server.rows]
        ))
        return SimResult(
            completions=dict(self.completions), end_s=self.now, host_j=self.host_j,
            phone_j=self.phone_j, state_s=dict(self.state_s), loads=self.server.loads,
            provisions=sum(p.provisions for p in self.phones.values()),
            unfinished=unfinished, decisions=list(self.decisions),
        )


def _describe(action: Action) -> str:
    if isinstance(action, Admit):
        mode = "" if action.assisted is None else (" assisted" if action.assisted else " desktop")
        return "admit " + ",".join(action.request_ids) + mode
    if isinstance(action, Park):
        return "park " + ",".join(action.request_ids)
    if isinstance(action, Switch):
        return "switch " + action.model
    if isinstance(action, Provision):
        return "provision " + action.phone + " " + action.model
    if isinstance(action, SetAssist):
        return "assist " + ("on" if action.on else "off")
    if isinstance(action, DeferProvision):
        return "defer " + action.phone + " " + str(action.model)
    return repr(action)


# Policies --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SequentialPolicy:
    """Emulation of the frozen rule cascade (paper_config_v1).

    ``serialize_assisted_joiners``: a same-model arrival after the cohort window waits for an assisted
    batch to finish (the capacity-1 phone lanes are held); while the batch is host-only it joins at
    once. ``retry_reprovision_on_release``: a re-provision deferred because the phone was in use or
    thermally excluded is retried when the lease is released or the exclusion ends (the frozen code's
    release re-evaluation plus WS1's ``event_replanning`` for admissibility changes); without it a
    deferred target is only re-evaluated at the next arrival or desktop switch (hypothetical, kept for
    sensitivity). Phone weight loads are refused while the phone is thermally excluded.
    ``affinity_refuses_serialized_join`` reproduces the pre-09-28 code, where an arrival that would
    only be serialized behind the assisted holder was left behind the queued switch.
    """

    helpers: bool = True
    cohort_window_s: float = 2.5
    serialize_assisted_joiners: bool = True
    model_affinity: bool = True
    affinity_maximum_bypasses: int = 10
    affinity_maximum_wait_s: float = 1200.0
    residency_hysteresis_s: float = 20.0
    residency_hysteresis_min_probability: float = 0.5
    early_reprovision: bool = True
    retry_reprovision_on_release: bool = True
    late_adoption: bool = True
    affinity_refuses_serialized_join: bool = False

    def decide(self, sim: Simulator) -> list[Action]:
        actions: list[Action] = []
        if self.helpers and self.late_adoption:
            actions.extend(self._late_adoption(sim))
        actions.extend(self._admissions(sim))
        if self.helpers:
            actions.extend(self.provisioning(sim, actions))
        return actions

    # admission and switching

    def _protected(self, sim: Simulator, request: SimRequest) -> bool:
        return (
            sim.bypasses.get(request.request_id, 0) >= self.affinity_maximum_bypasses
            or sim.now - request.arrival_s >= self.affinity_maximum_wait_s
        )

    def _blocked(self, sim: Simulator, request: SimRequest) -> bool:
        """Whether an older other-model request keeps ``request`` behind its switch."""
        older = [q for q in sim.queue if q.model != request.model and q.arrival_s < request.arrival_s]
        if not older:
            return False
        if not self.model_affinity:
            return True
        return any(self._protected(sim, q) for q in older)

    def _admissions(self, sim: Simulator) -> list[Action]:
        server = sim.server
        if server.loading is not None:
            return []
        if server.resident is not None:
            model = server.resident
            slots = sim.cost.model(model).slots
            free = slots - len(server.rows)
            same = [r for r in sim.queued(model) if not self._blocked(sim, r)]
            if server.rows:
                joiners = same[:max(free, 0)]
                if not joiners or not server.decoding():
                    return []
                in_window = (
                    server.phase_start_s is not None
                    and sim.now - server.phase_start_s <= self.cohort_window_s + EPSILON_S
                )
                if not server.assisted or in_window or not self.serialize_assisted_joiners:
                    return [Admit(tuple(r.request_id for r in joiners))]
                if self.affinity_refuses_serialized_join and sim.queued_other_than(model):
                    return []
                return [Park(tuple(r.request_id for r in joiners))]
            if same:
                chosen = same[:slots]
                assisted = self.helpers and sim.helper_available(model)
                return [Admit(tuple(r.request_id for r in chosen), assisted)]
        others = sim.queued_other_than(server.resident)
        if server.rows or not others:
            return []
        target = min(others, key=lambda r: (r.arrival_s, r.request_id)).model
        if self._hysteresis_holds(sim, target):
            return [Wake(server.last_release_s + self.residency_hysteresis_s)]
        return [Switch(target)]

    def _hysteresis_holds(self, sim: Simulator, target: str) -> bool:
        server = sim.server
        window = self.residency_hysteresis_s
        if not window or server.resident is None or server.last_release_s is None:
            return False
        if server.released_model != server.resident or sim.now >= server.last_release_s + window:
            return False
        oldest = min(r.arrival_s for r in sim.queued(target))
        if sim.now - oldest > window:
            return False
        probability = sim.arrival_probability(server.resident, window)
        return probability is not None and probability >= self.residency_hysteresis_min_probability

    # phone helpers

    def _late_adoption(self, sim: Simulator) -> list[Action]:
        server = sim.server
        rows = server.decoding()
        if not rows or server.assisted or not sim.helper_available(server.resident):
            return []
        if max(row.tokens_left for row in rows) < sim.cost.late_adoption_minimum_tokens:
            return []
        return [SetAssist(True)]

    def commitment(self, sim: Simulator, pending: Sequence[Action] = ()) -> str | None:
        """The desktop commitment the phone follows: loading > executing > next queued > resident."""
        for action in pending:
            if isinstance(action, Switch):
                return action.model
        server = sim.server
        if server.loading is not None:
            return server.loading
        if server.rows:
            return server.resident
        for action in pending:
            if isinstance(action, Admit):
                return server.resident
        if server.resident is not None and sim.queued(server.resident):
            return server.resident
        if sim.queue:
            return min(sim.queue, key=lambda r: (r.arrival_s, r.request_id)).model
        return server.resident

    def provisioning(self, sim: Simulator, pending: Sequence[Action]) -> list[Action]:
        if not sim.options.phones_enabled:
            return []
        name = sim.cost.primary_phone
        phone = sim.phones[name]
        target = self.commitment(sim, pending)
        if target is None or not sim.cost.model(target).assistable:
            return []
        if phone.model == target or phone.loading == target:
            return [DeferProvision(name, None)] if phone.deferred is not None else []
        if not self._worth_provisioning(sim, target, pending):
            return []
        switching = any(isinstance(a, Switch) for a in pending)
        trigger = (
            switching and self.early_reprovision
            or sim.just_arrived()
            or phone.deferred is not None and self.retry_reprovision_on_release
            or phone.model is None and phone.loading is None
        )
        if not trigger:
            return []
        if not sim.phone_free(name) or not sim.thermal_ok(name):
            return [] if phone.deferred == target else [DeferProvision(name, target)]
        return [Provision(name, target)]

    def _worth_provisioning(self, sim: Simulator, target: str, pending: Sequence[Action]) -> bool:
        spec = sim.cost.phones[sim.cost.primary_phone]
        server = sim.server
        remaining = sim.remaining_tokens(target)
        if server.resident == target and server.decoding() and not any(isinstance(a, Switch) for a in pending):
            step = sim.cost.model(target).step_s(len(server.decoding()), False)
            longest = max(row.tokens_left for row in server.decoding())
            queued = sum(r.output_tokens for r in sim.queued(target))
            remaining = max(0.0, longest - spec.provision_s / step) + queued
        return remaining >= sim.cost.late_adoption_minimum_tokens


@dataclass(frozen=True)
class LegacyPolicy:
    """All-desktop arrival order (the legacy dispatcher).

    The request that triggers a model load runs alone after the load; afterwards the same-model
    requests at the head of the queue start together (legacy 004+005 and 009+010).
    """

    def decide(self, sim: Simulator) -> list[Action]:
        server = sim.server
        if server.loading is not None or server.rows or not sim.queue:
            return []
        head = min(sim.queue, key=lambda r: (r.arrival_s, r.request_id))
        if server.resident != head.model:
            return [Switch(head.model, owner=head.request_id)]
        if server.load_owner == head.request_id:
            return [Admit((head.request_id,), False)]
        batch = []
        for request in sorted(sim.queue, key=lambda r: (r.arrival_s, r.request_id)):
            if request.model != head.model:
                break
            batch.append(request.request_id)
        slots = sim.cost.model(head.model).slots
        return [Admit(tuple(batch[:slots]), False)]


# Metrics ---------------------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 100])."""
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(result: SimResult) -> dict[str, float]:
    latencies = [c.latency_s for c in result.completions.values()]
    return {
        "host_kj": result.host_j / 1e3,
        "fleet_kj": result.fleet_j / 1e3,
        "end_s": result.end_s,
        "p50_s": percentile(latencies, 50),
        "p90_s": percentile(latencies, 90),
        "loads": result.loads,
        "provisions": result.provisions,
        "assisted_token_share": (
            sum(c.assisted_tokens for c in result.completions.values())
            / max(1, sum(c.output_tokens for c in result.completions.values()))
        ),
    }


def simulate(
    requests: Sequence[SimRequest],
    policy: Policy,
    cost: CostModel,
    options: SimOptions = SimOptions(),
) -> SimResult:
    return Simulator(requests, cost, options).run(policy)
