"""Measured cost model and constraints for the joint (bounded-horizon) planner.

Times are seconds, powers watts, energies joules. Every constant comes from recorded two-phone
``longtail_eval_v2`` runs (s1a-s2a, 2026-09-28/29) or from the measurements quoted in
``research_dev/talks.md``; ``PROVENANCE`` names the source of each group, and
``campaigns/burstgpt/joint_planner_eval.py`` re-derives the run-derived ones from the compact run
summaries in ``tests/data/joint_planner/eval_v2_runs.json``.

Host power is CPU package plus GPU board. Phone power is the scheduler's assumed model (4.5 W active,
0.875 W idle, evidence ``ASSUMED_4P5W``), not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping


JOINT_PLANNER_MODEL_SCHEMA = "research-scheduler-joint-planner-cost-model-v1"


class JointPlannerModelError(ValueError):
    pass


@dataclass(frozen=True)
class ModelCosts:
    """One desktop model: server slots, switch cost, prefill and per-step decode cost by batch.

    ``desktop_step_s[b - 1]`` / ``assisted_step_s[b - 1]`` are decode step periods at batch ``b``
    (one token per row per step); the power tuples are the host power in that state.
    ``assisted_step_s`` is empty when no phone holds shards of this model. A load costs
    ``load_fixed_j + load_power_w * load_s`` (measured load energy grows slower than duration).
    """

    slots: int
    load_s: float
    load_power_w: float
    prefill_fixed_s: float
    prefill_s_per_token: float
    desktop_step_s: tuple[float, ...]
    desktop_power_w: tuple[float, ...]
    assisted_step_s: tuple[float, ...] = ()
    assisted_power_w: tuple[float, ...] = ()
    helper_devices: tuple[str, ...] = ()
    load_fixed_j: float = 0.0

    def __post_init__(self) -> None:
        if type(self.slots) is not int or self.slots < 1:
            raise JointPlannerModelError("model slots must be a positive integer")
        for name in ("desktop_step_s", "desktop_power_w"):
            if len(getattr(self, name)) != self.slots:
                raise JointPlannerModelError(name + " needs one entry per slot")
        if len(self.assisted_step_s) != len(self.assisted_power_w):
            raise JointPlannerModelError("assisted step and power tuples differ in length")
        if self.assisted_step_s and len(self.assisted_step_s) != self.slots:
            raise JointPlannerModelError("assisted_step_s needs one entry per slot")
        if bool(self.assisted_step_s) != bool(self.helper_devices):
            raise JointPlannerModelError("assisted costs require helper devices and vice versa")
        values = (
            self.load_s, self.load_power_w, self.load_fixed_j, self.prefill_fixed_s, self.prefill_s_per_token,
            *self.desktop_step_s, *self.desktop_power_w, *self.assisted_step_s, *self.assisted_power_w,
        )
        if any(not isinstance(v, (int, float)) or v < 0 for v in values):
            raise JointPlannerModelError("model costs must be nonnegative numbers")

    @property
    def assistable(self) -> bool:
        return bool(self.assisted_step_s)

    def step_s(self, batch: int, assisted: bool) -> float:
        table = self.assisted_step_s if assisted else self.desktop_step_s
        return table[min(max(batch, 1), self.slots) - 1]

    def power_w(self, batch: int, assisted: bool) -> float:
        table = self.assisted_power_w if assisted else self.desktop_power_w
        return table[min(max(batch, 1), self.slots) - 1]

    def prefill_s(self, input_tokens: int) -> float:
        return self.prefill_fixed_s + self.prefill_s_per_token * input_tokens

    def load_energy_j(self, duration_s: float | None = None) -> float:
        return self.load_fixed_j + self.load_power_w * (self.load_s if duration_s is None else duration_s)


@dataclass(frozen=True)
class PhoneCosts:
    """A helper phone: shard sessions re-provisioned one after another, assumed power."""

    sessions: int
    session_load_s: float
    active_power_w: float = 4.5
    idle_power_w: float = 0.875
    fixed_model: str | None = None

    @property
    def provision_s(self) -> float:
        return self.sessions * self.session_load_s


@dataclass(frozen=True)
class CostModel:
    models: Mapping[str, ModelCosts]
    phones: Mapping[str, PhoneCosts]
    primary_phone: str
    idle_loaded_w: float
    idle_unloaded_w: float
    late_adoption_minimum_tokens: int = 24
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.primary_phone not in self.phones:
            raise JointPlannerModelError("primary phone is not a phone")
        for name, costs in self.models.items():
            unknown = set(costs.helper_devices) - set(self.phones)
            if unknown:
                raise JointPlannerModelError(name + " names unknown helper devices")
            if costs.assistable and self.primary_phone not in costs.helper_devices:
                raise JointPlannerModelError(name + " assistance needs the primary phone")

    def model(self, name: str) -> ModelCosts:
        try:
            return self.models[name]
        except KeyError:
            raise JointPlannerModelError("unknown model " + name) from None

    def best_token_energy_j(self, name: str) -> float:
        """Lowest host+phone energy per output token over batch sizes and placements."""
        costs = self.model(name)
        best = min(
            costs.step_s(b, False) * costs.power_w(b, False) / b for b in range(1, costs.slots + 1)
        )
        if costs.assistable:
            phones = sum(self.phones[d].active_power_w for d in costs.helper_devices)
            best = min(best, min(
                costs.step_s(b, True) * (costs.power_w(b, True) + phones) / b
                for b in range(1, costs.slots + 1)
            ))
        return best

    def with_models(self, **models: ModelCosts) -> "CostModel":
        return replace(self, models=MappingProxyType({**self.models, **models}))

    def to_json(self) -> dict[str, object]:
        return {
            "schema": JOINT_PLANNER_MODEL_SCHEMA,
            "models": {
                name: {
                    "slots": c.slots, "load_s": c.load_s, "load_power_w": c.load_power_w,
                    "load_fixed_j": c.load_fixed_j,
                    "prefill_fixed_s": c.prefill_fixed_s, "prefill_s_per_token": c.prefill_s_per_token,
                    "desktop_step_s": list(c.desktop_step_s), "desktop_power_w": list(c.desktop_power_w),
                    "assisted_step_s": list(c.assisted_step_s), "assisted_power_w": list(c.assisted_power_w),
                    "helper_devices": list(c.helper_devices),
                }
                for name, c in sorted(self.models.items())
            },
            "phones": {
                name: {
                    "sessions": p.sessions, "session_load_s": p.session_load_s,
                    "active_power_w": p.active_power_w, "idle_power_w": p.idle_power_w,
                    "fixed_model": p.fixed_model,
                }
                for name, p in sorted(self.phones.items())
            },
            "primary_phone": self.primary_phone,
            "idle_loaded_w": self.idle_loaded_w,
            "idle_unloaded_w": self.idle_unloaded_w,
            "late_adoption_minimum_tokens": self.late_adoption_minimum_tokens,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class PlannerConstraints:
    """Feasibility rules every planned schedule must meet.

    ``maximum_latency_ppm``: a request's arrival-to-completion time may be at most this multiple of
    its desktop-only estimate (the same bound the adaptive controller applies per token; 1,250,000 =
    1.25x). The desktop-only estimate is the sequential policy's completion with every helper off,
    from the same state. When the sequential policy itself exceeds the bound for a request, that
    request is only required to finish no later than the sequential plan.
    ``maximum_displacement_s``: no request may start more than this many seconds later than in the
    sequential plan (bounded queue displacement).
    ``maximum_epoch_delay_s``: at every decision epoch no known request may finish more than this
    many seconds later than under the sequential plan from the same state (keeps each deviation a
    near Pareto improvement; the two bounds above cap the drift across epochs because they are
    anchored when the planner first sees the request).
    ``step_latency_ppm``: an assisted batch's step may be at most this multiple of the desktop step
    at the same batch size.
    """

    maximum_latency_ppm: int = 1_250_000
    maximum_displacement_s: float = 120.0
    maximum_epoch_delay_s: float = 30.0
    step_latency_ppm: int = 1_250_000

    def __post_init__(self) -> None:
        for name in ("maximum_latency_ppm", "step_latency_ppm"):
            value = getattr(self, name)
            if type(value) is not int or value < 1_000_000:
                raise JointPlannerModelError(name + " must be an integer >= 1000000")
        if self.maximum_displacement_s < 0 or self.maximum_epoch_delay_s < 0:
            raise JointPlannerModelError("maximum_displacement_s and maximum_epoch_delay_s must be nonnegative")


PROVENANCE = MappingProxyType({
    "desktop_step_s": (
        "Qwen 611 ms at batch 1 and 615-632 ms at batch 4 (tp2 utilization audit, talks.md 2026-09-27); "
        "Qwen batch 2 = s2a 005/007 decode periods 624-628 ms (host-only pair); Gemma 464 ms "
        "(tp2 audit; s1d 002/006 host-only pair 463-465 ms)"
    ),
    "assisted_step_s": (
        "Gemma OP15 24 layers: s2a solo periods 387-410 ms, s2a 002/006 pair 400-402 ms; "
        "Qwen OP15 18 + Pixel 6 layers: s1c/s1d/s2a solo 483-499 ms, pairs s1a/s1d/s2a 509-521 ms, "
        "s1c 4-row batch 655-705 ms (phone calls rows=4: OP15 14.3 vs 10.5 ms, Pixel 22.7 vs 13.1 ms)"
    ),
    "power": (
        "host power while decoding = (execution-window energy - prefill at the host-only batch-1 power) "
        "/ decode time (RESULT execution_receipt energy, s1a-s2a): Gemma assisted 50.1 W solo (n=9), "
        "58.3 W pair (n=4); Qwen assisted 45.6 W solo (n=2), 47.4 W pair (n=6), 51.0 W 4-row (s1c, "
        "n=1), 3 rows interpolated; host-only: Qwen pair 123 W (s2a 005/007), Gemma pair 107 W over a "
        "window that was 25 % assisted (s1d 002/006) -> 123 W host-only; batch 1 = 122 W implied by the "
        "all-desktop legacy total (228.5 kJ / 2,244 s after 425 s of loads and 59 s idle)"
    ),
    "load": (
        "desktop model load (15+15+5 transition receipts, s1a-s2a): mean duration Gemma 51.2 s, Qwen "
        "74.2 s, Llama 7.3 s (Llama evicts cuda0 residency); energy least-squares E = a + b x d: Gemma "
        "1027 J + 9.4 W, Qwen 1021 J + 13.3 W, Llama 40 J + 29.1 W (residual sd 113 / 99 / 35 J)"
    ),
    "prefill": "linear fit over s1a-s2a first-token minus execution start (<15 s): Gemma 3.11 s + "
               "3.36 ms/token, Qwen 3.56 s + 6.52 ms/token",
    "phone_provision": "OP15 3 HTP sessions re-provisioned in sequence, SESSION_LOADING -> SESSION_READY "
                       "8.8-40 s each, mean 14 s (s1a-s2a phone_residency_events); Pixel holds Qwen "
                       "layers 18-23 for the whole run",
    "idle": "loaded idle 22 W = residual of s1a/s2a totals after loads and execution windows (tp2 GPU "
            "loaded-idle 27.6 W, dp1 IDLE_MIN 18.5 W); unloaded 15.4 W (tp2 audit floor)",
})


def measured_eval_v2_cost_model() -> CostModel:
    """The frozen cost model for the 4060 Ti + OP15 + Pixel rig and longtail_eval_v2."""
    phones = MappingProxyType({
        "op15": PhoneCosts(sessions=3, session_load_s=14.0),
        "pixel": PhoneCosts(sessions=1, session_load_s=0.0, fixed_model="qwen"),
    })
    models = MappingProxyType({
        "gemma": ModelCosts(
            slots=2, load_s=51.2, load_power_w=9.4, load_fixed_j=1027.0,
            prefill_fixed_s=3.11, prefill_s_per_token=0.00336,
            desktop_step_s=(0.464, 0.465), desktop_power_w=(120.0, 123.0),
            assisted_step_s=(0.398, 0.401), assisted_power_w=(50.1, 58.3),
            helper_devices=("op15",),
        ),
        "qwen": ModelCosts(
            slots=4, load_s=74.2, load_power_w=13.3, load_fixed_j=1021.0,
            prefill_fixed_s=3.56, prefill_s_per_token=0.00652,
            desktop_step_s=(0.611, 0.626, 0.629, 0.632), desktop_power_w=(120.0, 123.0, 124.0, 125.0),
            assisted_step_s=(0.489, 0.514, 0.595, 0.680), assisted_power_w=(45.6, 47.4, 49.2, 51.0),
            helper_devices=("op15", "pixel"),
        ),
        "llama": ModelCosts(
            slots=1, load_s=7.3, load_power_w=29.1, load_fixed_j=40.0,
            prefill_fixed_s=0.1, prefill_s_per_token=0.0,
            desktop_step_s=(0.005,), desktop_power_w=(60.0,),
        ),
    })
    return CostModel(
        models=models, phones=phones, primary_phone="op15",
        idle_loaded_w=22.0, idle_unloaded_w=15.4,
        provenance=PROVENANCE,
    )
