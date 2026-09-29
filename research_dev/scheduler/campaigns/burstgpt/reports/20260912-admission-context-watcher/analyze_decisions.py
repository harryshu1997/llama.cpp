"""Per-request assistance decisions for a many-request run.

usage: analyze_decisions.py <run dir>
Prints, per large-model request: layout generation, evidence state, the
first decision reason, the zero-assistance reasons seen, assisted token
share and the verification outcome; then role totals.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def main(run: Path) -> int:
    result = json.load(open(run / "RESULT.json"))
    store = json.load(open(run / "ADAPTIVE_DECODE_OBSERVATIONS.json"))
    events = result["request_helper_events"]
    roles = {value: key for key, value in result["model_roles"].items()}
    groups = {group["request_id"]: group for group in store["groups"]}
    totals = collections.defaultdict(lambda: [0, 0])
    print(f"{'idx':>4} {'role':<6} {'gen':>4} {'evid':<9} {'tok':>4} {'assist%':>7} {'first reason':<26} {'verification':<22} zero-assistance reasons")
    for row in sorted(result["request_results"], key=lambda r: r["replay_arrival_us"]):
        role = roles.get(row["model_id"], "?")
        if role == "llama":
            continue
        rid = row["request_id"]
        decisions = [e for e in events if e["request_id"] == rid and e["kind"] == "ASSISTANCE_DECISION"]
        first = decisions[0] if decisions else {}
        zero = collections.Counter(e.get("reason") for e in decisions if e.get("selected_fraction_ppm") == 0)
        group = groups.get(rid)
        assisted = 0
        if group:
            for w in group["windows"]:
                if (w.get("completed_phone_calls") or 0) > 0 and w["policy"]["split_fraction_ppm"] > 0:
                    assisted += w["token_end"] - w["token_start"]
        tokens = row["output_tokens"]
        totals[role][0] += tokens
        totals[role][1] += assisted
        verification = (decisions[-1].get("verification") or {}) if decisions else {}
        print(f"{row['combined_request_index']:>4} {role:<6} {str(first.get('helper_layout_generation')):>4} {str(first.get('helper_evidence_state')):<9} {tokens:>4} {100 * assisted / max(1, tokens):7.1f} "
              f"{str(first.get('reason')):<26} {str(verification.get('outcome')) + '/' + str(verification.get('reason'))[:12]:<22} {dict(zero)}")
    for role, (tokens, assisted) in sorted(totals.items()):
        print(f"{role}: {assisted}/{tokens} tokens assisted ({100 * assisted / max(1, tokens):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
