#!/usr/bin/env python3
"""S10-V0 final verdict, computed from the CP4 gate summary and the CP2 measured
mechanism-availability facts, applying the PLAN/NEXT_PLAN gate logic. Deterministic.

Verdict domain (exactly one):
  PASS                          - opportunity + causal + mechanism + physical all pass
  MECHANISM_PASS_ENERGY_BLOCKED - analytic mechanism passes but physical energy blocked
  FAIL                          - the first required analytic/real gate fails
"""
import json
import os
import sys

SP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(SP, "artifacts")

gate = json.load(open(os.path.join(ART, "cp4_gate_summary.json")))
avail = gate["measured_mechanism_availability"]

# CP0 physical boundary
physical_gate = "ENERGY_BLOCKED"   # from artifacts/cp0_integrity.txt

# Opportunity gate: >=2 adjacent load bins with C4>=15% and SLO no worse
opp_hits = gate["opportunity_gate"]["hits"]
opportunity_met_somewhere = len(opp_hits) > 0
opportunity_robust = gate["opportunity_gate"]["any_conservative_hit"]  # holds under conservative power

# Mechanism gate: a MEASURED larger server batch / lower cap / break-even low-power
# interval must explain the benefit, AND C5 timely phone islands > C2.
lower_state = avail["lower_power_state_below_idle"]           # measured False
cap_settable = avail["power_cap_settable"]                    # measured False
any_larger_batch = avail["any_C4_creates_larger_server_batch"]  # measured False
causal_explanation_available = bool(lower_state or cap_settable or any_larger_batch)
# C5 timely > C2 in every opportunity hit?
c5_timely_gt_c2_in_hits = all(all(h["C5_timely_gt_C2"]) for h in opp_hits) if opp_hits else False
mechanism_gate_met = causal_explanation_available and c5_timely_gt_c2_in_hits

reasons = []
if physical_gate == "ENERGY_BLOCKED":
    reasons.append("PHYSICAL: ENERGY_BLOCKED (CP0) - no synchronized wall meter; GPU-board sensor "
                   "only, ~1.5 Hz; power caps unsettable; no phone charger meter -> cannot resolve 10%.")
if not opportunity_robust:
    reasons.append("OPPORTUNITY not robust: C4 >= 15% appears ONLY under the mechanism-FAVORABLE "
                   "(blocked, unmeasured) phone/USB power AND a degenerate all-lone-weight slack cohort; "
                   "under the conservative measured-plausible power model C4 = +0.0% in EVERY bin "
                   "(the perfect oracle never offloads).")
if not causal_explanation_available:
    reasons.append("MECHANISM gate FAILS on measured hardware: none of the three certified causal "
                   "levers is available/triggered - no A6000 power state below auto-P8 (25 W); power "
                   "caps not settable (root); and offload NEVER creates a larger server batch "
                   "(any_C4_creates_larger_server_batch=False, offload only removes work). The apparent "
                   "favorable-power gain is skipped-GPU-us per-island offload substitution, which the "
                   "design explicitly excludes and grants no energy credit.")

# Decision
if physical_gate == "READY" and opportunity_robust and mechanism_gate_met:
    verdict = "PASS"
elif opportunity_robust and mechanism_gate_met and physical_gate != "READY":
    verdict = "MECHANISM_PASS_ENERGY_BLOCKED"
else:
    verdict = "FAIL"

out = {
    "verdict": verdict,
    "physical_gate": physical_gate,
    "opportunity_met_under_favorable_only": opportunity_met_somewhere and not opportunity_robust,
    "opportunity_robust": opportunity_robust,
    "mechanism_causal_explanation_available": causal_explanation_available,
    "mechanism_gate_met": mechanism_gate_met,
    "measured_facts": {
        "lower_power_state_below_idle": lower_state,
        "power_cap_settable": cap_settable,
        "any_C4_creates_larger_server_batch": any_larger_batch,
    },
    "opportunity_hits": opp_hits,
    "reasons": reasons,
}
with open(os.path.join(ART, "cp_verdict.json"), "w") as f:
    json.dump(out, f, indent=2, sort_keys=True)
print(json.dumps(out, indent=2, sort_keys=True))
sys.exit(0)
