"""Decode a saved host call gap; this does not execute inference."""

import argparse
import hashlib
import json
from pathlib import Path
import re


parser = argparse.ArgumentParser()
parser.add_argument("stderr", type=Path)
parser.add_argument("--from-call", type=int, required=True)
parser.add_argument("--to-call", type=int, required=True)
args = parser.parse_args()
raw = args.stderr.read_bytes()
lines = raw.decode("utf-8").splitlines()
calls = {}
tokens = []
warmups = []
clock = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) ")
for line_number, line in enumerate(lines, 1):
    if line.startswith("S41SERVERFFNUSB "):
        fields = dict(item.split("=", 1) for item in line.split()[1:])
        row = {key: int(value) for key, value in fields.items()}
        row["line_number"] = line_number
        calls[row["request"]] = row
    match = clock.match(line)
    if match is None:
        continue
    minute, second, millis, micros = map(int, match.groups())
    at_us = ((minute * 60 + second) * 1000 + millis) * 1000 + micros
    token = re.search(r"n_decoded = (\d+),", line)
    if token is not None:
        tokens.append({"token_index": int(token[1]), "server_log_us": at_us,
                       "line_number": line_number})
    if "CUDA graph warmup complete" in line:
        warmups.append({"server_log_us": at_us, "line_number": line_number})
left, right = calls[args.from_call], calls[args.to_call]
assert args.to_call == args.from_call + 1
assert left["line_number"] < right["line_number"]
between = [row for row in warmups
           if left["line_number"] < row["line_number"] < right["line_number"]]
before_token = max((row for row in tokens
                    if row["line_number"] < left["line_number"]),
                   key=lambda row: row["line_number"])
after_token = min((row for row in tokens
                   if row["line_number"] > left["line_number"]),
                  key=lambda row: row["line_number"])
assert between and after_token["line_number"] < right["line_number"]
print(json.dumps({
    "schema": "s42-saved-host-gap-diagnosis-v1",
    "stderr_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    "from_call": left, "to_call": right,
    "host_between_rpc_gap_ns": right["started_ns"] - left["d2h_completed_ns"],
    "from_call_rpc_ns": left["d2h_completed_ns"] - left["started_ns"],
    "to_call_rpc_ns": right["d2h_completed_ns"] - right["started_ns"],
    "cuda_recapture_count_between_calls": len(between),
    "first_to_last_recapture_log_us": (
        between[-1]["server_log_us"] - between[0]["server_log_us"]
    ),
    "cuda_recapture_events": between,
    "from_token": before_token, "to_token": after_token,
    "token_latency_us": after_token["server_log_us"] - before_token["server_log_us"],
    "next_tokens": [row for row in tokens
                    if after_token["token_index"] < row["token_index"]
                    <= after_token["token_index"] + 5],
    "timing_note": "Durations use each source's own clock; log span is not exclusive CUDA capture time.",
}, sort_keys=True, indent=2))
