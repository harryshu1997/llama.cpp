#!/usr/bin/env python3

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


SCHEMA = "s38-server-cohort-v1"


class CohortError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="ascii") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CohortError(f"non-object at {path}:{line_number}")
            values.append(value)
    return values


def group_key(request: dict[str, Any]) -> tuple[str, int]:
    fields = request.get("source_fields")
    if not isinstance(fields, dict):
        raise CohortError("request has no source_fields")
    question_type = fields.get("question_type")
    evidence_count = fields.get("evidence_count")
    if not isinstance(question_type, str) or type(evidence_count) is not int:
        raise CohortError("request has invalid stratification fields")
    return question_type, evidence_count


def quotas(groups: dict[tuple[str, int], list[dict[str, Any]]], count: int) -> dict[tuple[str, int], int]:
    total = sum(len(values) for values in groups.values())
    result = {key: count * len(values) // total for key, values in groups.items()}
    remainder = count - sum(result.values())
    ranked = sorted(
        groups,
        key=lambda key: (-(count * len(groups[key]) % total), key),
    )
    for key in ranked[:remainder]:
        result[key] += 1
    return result


def select_evenly(values: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count == 0:
        return []
    if count > len(values):
        raise CohortError("stratum quota exceeds its population")
    ordered = sorted(values, key=lambda item: (
        item["output_tokens"],
        item["input_tokens"],
        hashlib.sha256(item["event_id"].encode("ascii")).hexdigest(),
    ))
    indexes = [min(len(ordered) - 1, math.floor((index + 0.5) * len(ordered) / count)) for index in range(count)]
    if len(set(indexes)) != count:
        raise CohortError("even selection produced duplicate indexes")
    return [ordered[index] for index in indexes]


def select(requests: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count < 1 or count > len(requests):
        raise CohortError("count is outside the source trace")
    groups: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for request in requests:
        groups[group_key(request)].append(request)
    allocation = quotas(groups, count)
    selected = [
        request
        for key in sorted(groups)
        for request in select_evenly(groups[key], allocation[key])
    ]
    selected.sort(key=lambda item: (item["t_us"], item["event_id"]))
    base = selected[0]["t_us"]
    result: list[dict[str, Any]] = []
    for source in selected:
        item = dict(source)
        item["t_us"] = source["t_us"] - base
        result.append(item)
    return result


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a deterministic S38 server cohort")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    requests = load_jsonl(source)
    selected = select(requests, args.count)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(output, "".join(canonical_json(item) + "\n" for item in selected))
    group_counts = collections.Counter(group_key(item) for item in selected)
    manifest = {
        "schema": SCHEMA,
        "source": {"path": source.name, "sha256": file_sha256(source), "records": len(requests)},
        "output": {"path": output.name, "sha256": file_sha256(output), "records": len(selected)},
        "selection": "proportional question-type/evidence-count quotas; within each stratum, evenly spaced by output and input length",
        "statistics": {
            "input_tokens": sum(item["input_tokens"] for item in selected),
            "output_tokens": sum(item["output_tokens"] for item in selected),
            "span_us": selected[-1]["t_us"],
            "groups": {
                f"{key[0]}:evidence-{key[1]}": value
                for key, value in sorted(group_counts.items())
            },
        },
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    atomic_text(manifest_path, canonical_json(manifest) + "\n")
    print(canonical_json(manifest["statistics"] | {
        "status": "S38_COHORT_SELECTED",
        "cohort_sha256": manifest["output"]["sha256"],
    }))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CohortError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"S38_COHORT_ERROR: {error}") from None
