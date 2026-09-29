#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


USB_BYTES_PER_SECOND = 465_000_000.0
RPC_FLOOR_MS = 0.235
TOP_K = 32
TOP_K_ENTRY_BYTES = 8
LM_REDUCE_46080_M1_MS = 0.243


def load_rows(path):
    rows = []
    with path.open() as source:
        for line in source:
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def index_projection(rows):
    return {
        (row["k"], row["n"], row["m"], row["matrices"]): row
        for row in rows
    }


def transfer_ms(byte_count):
    return byte_count / USB_BYTES_PER_SECOND * 1000.0


def projection_rpc_ms(row):
    return (row["compute_p50_ms"] + RPC_FLOOR_MS +
            transfer_ms(row["wire_input_f16_bytes"] +
                        row["wire_output_f16_bytes"]))


def gain_percent(baseline, candidate):
    return (baseline - candidate) / baseline * 100.0


def projection_analysis(root):
    cpu = index_projection(load_rows(root / "cpu_projection.jsonl"))
    cpu.update(index_projection(load_rows(root / "cpu_q_half.jsonl")))
    phone = index_projection(load_rows(root / "phone_projection_v2.raw"))
    phone.update(index_projection(load_rows(root / "phone_q_half.raw")))
    result = []
    for m in (1, 8, 32, 128, 512):
        cpu_q = cpu[(3840, 4096, m, 1)]["compute_p50_ms"]
        cpu_kv = cpu[(3840, 2048, m, 2)]["compute_p50_ms"]
        cpu_qkv = cpu[(3840, 2048, m, 4)]["compute_p50_ms"]
        cpu_q_half_remain = cpu[(3840, 2048, m, 3)]["compute_p50_ms"]
        cpu_o = cpu[(4096, 3840, m, 1)]["compute_p50_ms"]
        cpu_o_half = cpu[(2048, 3840, m, 1)]["compute_p50_ms"]

        qkv_candidates = {
            "cpu": cpu_qkv,
            "q_half": max(
                cpu_q_half_remain,
                projection_rpc_ms(phone[(3840, 2048, m, 1)])),
            "q_full": max(
                cpu_kv,
                projection_rpc_ms(phone[(3840, 4096, m, 1)])),
            "kv_full": max(
                cpu_q,
                projection_rpc_ms(phone[(3840, 2048, m, 2)])),
            "qkv_full": projection_rpc_ms(phone[(3840, 2048, m, 4)]),
        }
        o_candidates = {
            "cpu": cpu_o,
            "o_half": max(
                cpu_o_half,
                projection_rpc_ms(phone[(2048, 3840, m, 1)])),
            "o_full": projection_rpc_ms(phone[(4096, 3840, m, 1)]),
        }
        qkv_choice = min(qkv_candidates, key=qkv_candidates.get)
        o_choice = min(o_candidates, key=o_candidates.get)
        result.append({
            "m": m,
            "qkv_cpu_ms": cpu_qkv,
            "qkv_choice": qkv_choice,
            "qkv_choice_ms": qkv_candidates[qkv_choice],
            "qkv_gain_percent": gain_percent(
                cpu_qkv, qkv_candidates[qkv_choice]),
            "qkv_candidates_ms": qkv_candidates,
            "o_cpu_ms": cpu_o,
            "o_choice": o_choice,
            "o_choice_ms": o_candidates[o_choice],
            "o_gain_percent": gain_percent(cpu_o, o_candidates[o_choice]),
            "o_candidates_ms": o_candidates,
        })
    return result


def lm_analysis(root):
    cpu_rows = (load_rows(root / "cpu_lm_head.jsonl") +
                load_rows(root / "cpu_lm_head_tail.jsonl"))
    cpu = {(row["m"], row["n"]): row for row in cpu_rows}
    phone_rows = load_rows(root / "phone_lm_head_shards.raw")
    phone = {(row["m"], row["matrices"]): row for row in phone_rows}
    result = []
    vocab = 262144
    shard_rows = 32768
    for m in (1, 2, 4, 8):
        baseline = cpu[(m, vocab)]["compute_p50_ms"]
        candidates = {"cpu": baseline}
        details = {}
        one_shard_get_ms = phone[(m, 1)]["get_p50_ms"]
        for count in (1, 2, 3, 4, 6, 8):
            row = phone[(m, count)]
            remaining_rows = vocab - count * shard_rows
            cpu_branch = (cpu[(m, remaining_rows)]["compute_p50_ms"]
                          if remaining_rows else 0.0)
            output_copy_ms = one_shard_get_ms * count
            reduction_ms = (LM_REDUCE_46080_M1_MS *
                            (count * shard_rows / 46080.0) * m)
            wire_bytes = (row["wire_input_f16_bytes"] +
                          TOP_K * TOP_K_ENTRY_BYTES * m)
            phone_branch = (row["compute_p50_ms"] + output_copy_ms +
                            reduction_ms + RPC_FLOOR_MS +
                            transfer_ms(wire_bytes))
            combined = max(cpu_branch, phone_branch)
            name = f"phone_{count}_shards"
            candidates[name] = combined
            details[name] = {
                "offloaded_rows": count * shard_rows,
                "cpu_branch_ms": cpu_branch,
                "phone_branch_ms": phone_branch,
            }
        choice = min(candidates, key=candidates.get)
        result.append({
            "m": m,
            "cpu_ms": baseline,
            "choice": choice,
            "choice_ms": candidates[choice],
            "gain_percent": gain_percent(baseline, candidates[choice]),
            "choice_detail": details.get(choice),
            "candidates_ms": candidates,
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    output = {
        "assumptions": {
            "usb_MBps": USB_BYTES_PER_SECOND / 1e6,
            "rpc_floor_ms": RPC_FLOOR_MS,
            "lm_top_k": TOP_K,
            "lm_reduction_reference_ms": LM_REDUCE_46080_M1_MS,
        },
        "projection": projection_analysis(args.result_root),
        "lm_head": lm_analysis(args.result_root),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
