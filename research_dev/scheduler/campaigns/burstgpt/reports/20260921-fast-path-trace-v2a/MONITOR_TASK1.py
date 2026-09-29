"""Read live trace artifacts without modifying the rig or controlling processes."""

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys


def stream(path):
    tokens = []
    complete = False
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            try:
                value = json.loads(line[5:])
            except json.JSONDecodeError:
                continue
            tokens.extend(value.get("tokens", []))
            complete |= value.get("stop", False)
    return tokens, complete


def main(inputs, baseline):
    root = inputs / "run-treatment-1/run"
    reference = json.loads((baseline / "RESULT.json").read_text())
    completed = []
    differences = []
    progressing = []
    for row in reference["request_results"]:
        name = "streams/request-%03d.raw" % row["combined_request_index"]
        actual, stop = stream(root / name)
        expected, _ = stream(baseline / name)
        if stop and len(actual) == row["output_tokens"]:
            completed.append(row["request_id"])
            if actual != expected:
                differences.append(row["request_id"])
        elif actual:
            progressing.append({"request_id": row["request_id"], "tokens": len(actual),
                                "expected": row["output_tokens"]})
    calls = {}
    requests = Counter()
    for path in sorted(root.glob("*physical-hot-*.stderr")):
        contents = path.read_text()
        if not re.search(r"general\.architecture[^\n]*=\s*qwen3\b", contents):
            continue
        counts = Counter(int(value) for value in re.findall(
            r"S41SERVERFFNUSB[^\n]*\btokens=(\d+)\b", contents))
        if counts:
            calls[path.name] = dict(sorted(counts.items()))
        for contexts in re.findall(r"S41SERVERFFNCALL context=(\S+)", contents):
            for context in contexts.split(","):
                requests[bytes.fromhex(context.split(":")[0]).decode()] += 1
    result = {"at": datetime.now(timezone.utc).isoformat(), "inputs": str(inputs),
              "completed_streams": len(completed), "completed_request_ids": completed,
              "token_identical_complete_streams": len(completed) - len(differences),
              "output_differences": differences, "progressing_streams": progressing,
              "qwen_calls_by_rows": calls, "qwen_assisted_request_ids": sorted(requests),
              "qwen_multi_row_calls": sum(count for counts in calls.values()
                                           for width, count in counts.items() if width > 1)}
    snapshots = list((root / "snapshots").glob("*.scheduler.json"))
    if snapshots:
        latest = max(snapshots, key=lambda path: path.stat().st_mtime)
        try:
            snapshot = json.loads(latest.read_text())
        except json.JSONDecodeError:
            pass
        else:
            tickets = snapshot["runtime_controller"]["tickets"]
            result["latest_scheduler_snapshot"] = latest.name
            result["ticket_states"] = {key: value["dispatch_state"] for key, value in tickets.items()}
    for name in ("RESULT.json", "FAILURE.json"):
        path = root / name
        result[name] = path.exists()
        if path.exists():
            try:
                value = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            result[name + "_summary"] = {key: value[key] for key in (
                "status", "error", "counts", "duration_us", "trace_energy") if key in value}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
