"""Persist decoded changes between independently generated replay payloads."""

import collections
import difflib
import json
from pathlib import Path


def identity(value):
    if not isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    keys = ("event_kind", "kind", "event_type", "generation", "segment_id", "request_id", "session_id")
    return json.dumps({key: value[key] for key in keys if key in value}, sort_keys=True)


def decoded_diff(before, after, path=""):
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        return [row for key in sorted(before.keys() | after.keys())
                for row in decoded_diff(before.get(key), after.get(key), path + "/" + key)]
    if isinstance(before, list) and isinstance(after, list):
        if len(before) == len(after):
            return [row for i, (a, b) in enumerate(zip(before, after))
                    for row in decoded_diff(a, b, path + "/" + str(i))]
        rows = []
        matcher = difflib.SequenceMatcher(a=list(map(identity, before)), b=list(map(identity, after)), autojunk=False)
        for _, a, b, c, d in matcher.get_opcodes():
            for index in range(max(b - a, d - c)):
                old = before[a + index] if a + index < b else None
                new = after[c + index] if c + index < d else None
                rows.extend(decoded_diff(old, new, path + f"/{a + index}:{c + index}"))
        return rows
    return [{"path": path, "before": before, "after": after}]


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    old_hashes = json.loads((root / "replay-before/SHA256.json").read_text())
    new_hashes = json.loads((root / "replay-candidates-v2/SHA256.json").read_text())
    result = {}
    for case in sorted(old_hashes):
        old = json.loads((root / "replay-before" / (case + ".json")).read_text())
        new = json.loads((root / "replay-candidates-v2" / (case + ".json")).read_text())
        changes = decoded_diff(old, new)
        result[case] = {"old_sha256": old_hashes[case], "new_sha256": new_hashes[case], "changes": changes}
        print(case, len(changes), "changes", collections.Counter(row["path"].split("/")[1] for row in changes))
        print("non-hash fields:", [row["path"] for row in changes if not (
            isinstance(row["before"], str) and row["before"].startswith("sha256:")
            and isinstance(row["after"], str) and row["after"].startswith("sha256:"))])
    with (root / "REPLAY_DECODED_DIFF.json").open("x") as destination:
        json.dump(result, destination, indent=2, sort_keys=True)
        destination.write("\n")
