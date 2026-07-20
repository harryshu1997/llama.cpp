#!/usr/bin/env python3
"""S14 energy Stage C: mixed-workload A6000 GPU-board energy vs offload fraction.

Folds the MEASURED Stage A per-token tail energies onto the real mix-v1 workload,
at the deepest phone-FEASIBLE head depth certified by Stage B, to report the
mixed-workload A6000 GPU-board decode energy (and saving vs server-only) as a
function of how much of the generation decode is offloaded to the phone head.

Chain:
  Stage A (measured)  e_full, e_tail(k)   mJ/decoded-token on the A6000
  Stage B (measured)  max feasible head   [0,8) on OP15/HTP0 (k>=10 DSP-aborts)
  mix-v1  (real)      per-request output_tokens by service class
  =>  E_gpu(f) = total_tokens * [ f*e_tail(k*) + (1-f)*e_full ]
      saving(f) = f * (1 - e_tail(k*)/e_full)          f in [0, gen_token_fraction]

Scope: A6000 GPU_BOARD decode energy ONLY (same instrument/caveats as Stage A).
Phone/USB/total-wall energy is UNKNOWN. This is the overlap CEILING energy (it
assumes the A6000 stays busy while the phone runs the head; idle-wait is NOT
modelled). The offload fraction achievable at a given SLO is bounded below by the
Stage B phone-route latency; deeper than [0,8) needs a two-phone split.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
TRACE = SPIKE / "fixtures" / "mix_v1.trace.jsonl"
GEN_SERVICES = {"api_generation", "conversation_generation"}


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return "sha256:" + h.hexdigest()


def load_stage_a(path: Path) -> dict[int, float]:
    """Return {k: energy_per_token_mj_median} from a Stage A result."""
    d = json.loads(path.read_text())
    return {int(k): v["energy_per_token_mj_median"] for k, v in d["per_k"].items()}


def load_stage_b(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text())
    fs = d["feasibility_summary"]
    lat = {}
    for r in d["per_k"]:
        if r["placement_status"] == "SCHEDULED_PLACEMENT_OK":
            lat[r["k"]] = {"phone_head_us": r["stage_a_us_p50"], "server_tail_us": r["host_us_p50"]}
    return {"max_feasible_k": fs["max_feasible_head_k"], "latency": lat,
            "certified_k": fs["certified_k"], "dsp_abort_k": fs["dsp_abort_k"]}


def load_trace() -> list[dict[str, Any]]:
    return [json.loads(l) for l in TRACE.read_text().splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", default=str(HERE / "stage_a_k8_result.json"))
    ap.add_argument("--stage-b", default=str(HERE / "stageb_result.json"))
    ap.add_argument("--output", default=str(HERE / "stagec_result.json"))
    args = ap.parse_args()

    stage_a_path = Path(args.stage_a)
    stage_b_path = Path(args.stage_b)
    e = load_stage_a(stage_a_path)          # {k: mJ/tok}
    b = load_stage_b(stage_b_path)
    trace = load_trace()

    k_star = b["max_feasible_k"]
    if k_star not in e or 0 not in e:
        raise SystemExit(f"Stage A energy missing k=0 or k={k_star}; have {sorted(e)}")
    e_full = e[0]
    e_tail = e[k_star]
    s_star = 1.0 - e_tail / e_full          # measured per-token saving at k*

    # workload token split
    tok_by_service: dict[str, int] = defaultdict(int)
    for r in trace:
        tok_by_service[r["service"]] += int(r["output_tokens"])
    total_tokens = sum(tok_by_service.values())
    gen_tokens = sum(v for sv, v in tok_by_service.items() if sv in GEN_SERVICES)
    gen_frac = gen_tokens / total_tokens

    # E_gpu(f) with f = fraction of ALL decode tokens whose head runs on the phone.
    def e_gpu(f: float) -> float:
        return total_tokens * (f * e_tail + (1.0 - f) * e_full) / 1000.0  # Joules

    base_j = e_gpu(0.0)

    def saving(f: float) -> float:
        return 1.0 - e_gpu(f) / base_j      # == f * s_star

    # offload-fraction sweep + named operating points
    sweep = []
    fset = [0.0, 0.25, 0.5, gen_frac, 1.0]
    for f in sorted(set(round(x, 6) for x in fset)):
        sweep.append({
            "offload_token_fraction": f,
            "gpu_board_energy_j": e_gpu(f),
            "gpu_board_saving_vs_server_only": saving(f),
        })

    lat = b["latency"][k_star]
    phone_route_serial_us = lat["phone_head_us"] + lat["server_tail_us"]

    operating_points = {
        "server_only": {
            "offload_token_fraction": 0.0,
            "gpu_board_saving": 0.0,
            "note": "baseline: A6000 runs the full model for every token",
        },
        "offload_all_generation_to_phone_head": {
            "offload_token_fraction": gen_frac,
            "gpu_board_saving": saving(gen_frac),
            "note": f"route all {gen_tokens} generation tokens ({100*gen_frac:.1f}% of decode) "
                    f"to the phone head [0,{k_star}); rag_qa stays on the server",
            "slo_required": {
                "phone_route_latency_serial_us_per_request": phone_route_serial_us,
                "phone_head_us": lat["phone_head_us"],
                "server_tail_us": lat["server_tail_us"],
                "comment": "achievable only at a RELAXED SLO: the phone route is far slower "
                           "per request than server-only; overlap hides the phone head behind "
                           "independent server work, serial does not",
            },
        },
        "offload_all_decode_to_phone_head": {
            "offload_token_fraction": 1.0,
            "gpu_board_saving": saving(1.0),
            "note": f"ceiling if rag_qa generation ALSO used [0,{k_star}); == measured s(k*)",
        },
    }

    result = {
        "schema": "s14-stage-c-mixed-gpu-energy-v1",
        "scope": "A6000_GPU_BOARD_DECODE_ENERGY_ONLY_PHONE_UNKNOWN",
        "formal_claim": "MIXED_WORKLOAD_GPU_BOARD_ENERGY_OVERLAP_CEILING",
        "inputs": {
            "stage_a_result": str(stage_a_path.name),
            "stage_a_sha256": sha256_file(stage_a_path),
            "stage_b_result": str(stage_b_path.name),
            "stage_b_sha256": sha256_file(stage_b_path),
            "trace": str(TRACE.name),
            "trace_sha256": sha256_file(TRACE),
        },
        "offload_depth_k_star": k_star,
        "why_k_star": f"deepest Stage-B feasible single-phone head (certified {b['certified_k']}, "
                      f"DSP-abort {b['dsp_abort_k']}); [0,12) (22.8% Stage-A ceiling) needs a two-phone split",
        "measured_per_token_energy_mj": {"full_[0,48)": e_full, f"tail_[{k_star},48)": e_tail},
        "per_token_saving_at_k_star": s_star,
        "workload": {
            "total_output_tokens": total_tokens,
            "tokens_by_service": dict(tok_by_service),
            "generation_token_fraction": gen_frac,
        },
        "mixed_gpu_board_energy": {
            "server_only_energy_j": base_j,
            "offload_fraction_sweep": sweep,
            "operating_points": operating_points,
        },
        "headline": {
            "realisable_single_phone_gpu_board_saving_at_full_generation_offload":
                saving(gen_frac),
            "vs_ceiling_note": f"single-phone head caps at [0,{k_star}] = {100*s_star:.1f}% per-token; "
                               f"mixed workload realises {100*saving(gen_frac):.1f}% "
                               f"(generation is {100*gen_frac:.1f}% of decode). "
                               f"[0,12)=22.8% needs the two-phone split (Stage D).",
        },
        "caveats": [
            "A6000 GPU_BOARD decode energy ONLY (same instrument/caveats as Stage A). Phone/USB/total-wall UNKNOWN.",
            "Overlap CEILING: assumes the A6000 stays busy while the phone runs the head. Idle-wait energy is NOT modelled (the S11-E0 +53% failure mode).",
            "Offload fraction is a workload/scheduler knob; a given f is achievable only at an SLO >= the Stage B phone-route latency (relaxed-latency regime, as authorised).",
            "saving(f) = f * s(k*) is exact because A6000 power is flat (Stage A): energy scales with GPU-seconds = layers run.",
            "rag_qa (12.1% of tokens, model_class 'rag') is kept on the server here; its generation phase could also use the head, raising the ceiling toward s(k*).",
        ],
    }
    Path(args.output).write_text(json.dumps(result, indent=2))

    print(f"Stage C: mixed-workload A6000 GPU-board decode energy (k*=[0,{k_star}))")
    print(f"  measured: e_full={e_full:.1f} mJ/tok  e_tail[{k_star},48)={e_tail:.1f} mJ/tok  "
          f"s(k*)={100*s_star:.1f}%")
    print(f"  workload: {total_tokens} decode tokens, generation={100*gen_frac:.1f}%")
    print(f"  server-only baseline: {base_j:.0f} J")
    print("  offload fraction -> GPU-board saving:")
    for row in sweep:
        print(f"    f={row['offload_token_fraction']:.3f}  E={row['gpu_board_energy_j']:.0f} J  "
              f"saving={100*row['gpu_board_saving_vs_server_only']:.1f}%")
    print(f"  => realisable at full generation offload: "
          f"{100*saving(gen_frac):.1f}% A6000 decode energy "
          f"(phone route {phone_route_serial_us/1000:.0f} ms/req serial, needs relaxed SLO)")
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
