#!/usr/bin/env python3
"""Extended phone-head batch sweep (S14): push [0,8) past batch 32 to find the
throughput saturation point. Reuses stageb_headcert.run_one_k UNCHANGED (same
binaries, same placement-cert path). Records stage_a_us_p50 -> throughput and
watches for a DSP-queue abort / OOM ceiling as batch grows.

throughput_tok_s = batch * n_gen / (stage_a_us_p50 / 1e6)
forward_ms       = stage_a_us_p50 / n_gen / 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import stageb_headcert as sb

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="3C15AU002CL00000")
    ap.add_argument("--gpu-uuid", default="GPU-431a4567-fa90-7a73-625a-2ee6e7b5eaaf")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batches", default="48,64,96,128")
    ap.add_argument("--port", type=int, default=15579)
    ap.add_argument("--n-gen", type=int, default=8)
    # --driver-requests must be divisible by --driver-batch; requests/warmups are
    # set to the batch value per-batch (one warmup wave + one measured wave) unless
    # a positive multiple is passed explicitly.
    ap.add_argument("--requests", type=int, default=0)
    ap.add_argument("--warmups", type=int, default=0)
    ap.add_argument("--driver-context", type=int, default=512)
    ap.add_argument("--driver-max-prefill", type=int, default=64)
    ap.add_argument("--prompt", default="Explain why batching improves accelerator utilization.")
    ap.add_argument("--remote-dir", default="/data/local/tmp/ls-s14")
    ap.add_argument("--remote-bin", default="llama-layersplit")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--output", default=str(HERE / "batch_sweep_ext_result.json"))
    args = ap.parse_args()

    batches = [int(x) for x in args.batches.split(",")]
    log_dir = HERE / "logs_batch_ext"
    log_dir.mkdir(exist_ok=True)

    measured = []
    for b in batches:
        # driver-requests / driver-warmup must be multiples of the batch
        reqs = args.requests if args.requests > 0 else b
        warm = args.warmups if args.warmups > 0 else b
        print(f"\n###### k={args.k} batch={b} (requests={reqs} warmups={warm}) ######", flush=True)
        try:
            r = sb.run_one_k(args.serial, args.gpu_uuid, args.k, args.port, b, args.n_gen,
                             reqs, warm, args.driver_context,
                             args.driver_max_prefill, args.prompt, args.remote_dir,
                             args.remote_bin, log_dir, args.timeout)
        except Exception as e:  # noqa: BLE001
            print(f"  batch={b} FAILED: {e}", flush=True)
            measured.append({"batch": b, "status": "ERROR", "error": str(e)[:400]})
            # a hard failure at this batch is likely the ceiling; stop climbing
            break
        status = r.get("placement_status")
        p50 = r.get("stage_a_us_p50")
        if p50:
            tput = b * args.n_gen / (p50 / 1e6)
            fwd = p50 / args.n_gen / 1000.0
            print(f"  status={status} HTP0_ops={r.get('compute_htp0_nodes')} "
                  f"missing={r.get('missing_buffer_compute_nodes')} "
                  f"weight_mib={r.get('htp0_model_buffer_mib'):.0f} "
                  f"stage_a_p50={p50}us forward={fwd:.1f}ms tput={tput:.1f}tok/s", flush=True)
            measured.append({"batch": b, "status": status, "stage_a_us_p50": p50,
                             "forward_ms": round(fwd, 1), "throughput_tok_s": round(tput, 1),
                             "htp0_weight_mib": r.get("htp0_model_buffer_mib"),
                             "htp0_ops": r.get("compute_htp0_nodes"),
                             "missing_buffer_compute_nodes": r.get("missing_buffer_compute_nodes"),
                             "token_ids": r.get("token_ids")})
        else:
            print(f"  status={status} (no stage_a; likely DSP abort / infeasible)", flush=True)
            measured.append({"batch": b, "status": status or "NO_STAGE_A",
                             "infeasible_reason": r.get("infeasible_reason"),
                             "htp0_weight_mib": r.get("htp0_model_buffer_mib")})
            break

    out = {
        "schema": "s14-batch-sweep-ext-v1",
        "scope": "OP15_HTP0_BATCHED_HEAD_THROUGHPUT_ONLY_PHONE_ENERGY_UNKNOWN",
        "head": f"[0,{args.k})", "n_gen": args.n_gen, "flash_attn": "on",
        "device": {"serial": args.serial, "soc": "SM8850", "hexagon": "v81", "backend": "HTP0"},
        "note": "extends stage_d_batch_scaling.json (b1..b32) to find saturation; "
                "throughput = batch*n_gen/(stage_a_us_p50/1e6)",
        "measured": measured,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
