#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_MANIFEST = HERE / "CP0_R1_MMLU64_SOURCES_V2_2.json"
DEFAULT_CORPUS = HERE / "CP0_R1_MMLU64_CORPUS_V2_2.jsonl"

DATASET = "cais/mmlu"
REVISION = "bc5d09e5f0d160a95bcd36354bb5e16e50afe270"
SPLIT = "test"
SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(root: Path, subject: str) -> Path:
    return root / subject / "mmlu-test.parquet"


def build_source_manifest(root: Path) -> dict[str, Any]:
    files = []
    for subject in SUBJECTS:
        path = source_path(root, subject)
        if not path.is_file():
            raise RuntimeError(f"missing source: {path}")
        files.append(
            {
                "bytes": path.stat().st_size,
                "path": f"{subject}/mmlu-test.parquet",
                "sha256": sha256_file(path),
                "subject": subject,
            }
        )
    return {
        "dataset": DATASET,
        "files": files,
        "revision": REVISION,
        "schema": "s39-cp0-r1-mmlu-source-manifest-v2.2",
        "split": SPLIT,
        "subject_order": "ASCII_ASCENDING",
    }


def load_source_manifest(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if canonical_bytes(value) != raw:
        raise RuntimeError("source manifest is not canonical")
    if set(value) != {
        "dataset",
        "files",
        "revision",
        "schema",
        "split",
        "subject_order",
    }:
        raise RuntimeError("source manifest fields differ")
    if value["schema"] != "s39-cp0-r1-mmlu-source-manifest-v2.2":
        raise RuntimeError("source manifest schema differs")
    if value["dataset"] != DATASET or value["revision"] != REVISION:
        raise RuntimeError("source manifest dataset differs")
    if value["split"] != SPLIT or value["subject_order"] != "ASCII_ASCENDING":
        raise RuntimeError("source manifest selection differs")
    if len(value["files"]) != len(SUBJECTS):
        raise RuntimeError("source manifest file count differs")
    for subject, record in zip(SUBJECTS, value["files"]):
        if set(record) != {"bytes", "path", "sha256", "subject"}:
            raise RuntimeError(f"source fields differ: {subject}")
        if record["subject"] != subject:
            raise RuntimeError(f"source order differs: {subject}")
        if record["path"] != f"{subject}/mmlu-test.parquet":
            raise RuntimeError(f"source path differs: {subject}")
        if type(record["bytes"]) is not int or record["bytes"] <= 0:
            raise RuntimeError(f"source size differs: {subject}")
        if (
            type(record["sha256"]) is not str
            or len(record["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in record["sha256"])
        ):
            raise RuntimeError(f"source digest differs: {subject}")
    return value


def verify_sources(root: Path, manifest: dict[str, Any]) -> None:
    for record in manifest["files"]:
        path = root / record["path"]
        if path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"source size changed: {record['subject']}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"source digest changed: {record['subject']}")


def load_rows(root: Path, manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    import pyarrow.parquet as parquet

    result = {}
    for record in manifest["files"]:
        table = parquet.read_table(root / record["path"])
        if table.column_names != ["question", "choices", "answer"]:
            raise RuntimeError(f"source schema changed: {record['subject']}")
        rows = table.to_pylist()
        if not rows:
            raise RuntimeError(f"empty source: {record['subject']}")
        for index, row in enumerate(rows):
            if set(row) != {"answer", "choices", "question"}:
                raise RuntimeError(f"row fields changed: {record['subject']}:{index}")
            if type(row["question"]) is not str or not row["question"]:
                raise RuntimeError(f"question changed: {record['subject']}:{index}")
            if (
                type(row["choices"]) is not list
                or len(row["choices"]) != 4
                or any(type(choice) is not str for choice in row["choices"])
            ):
                raise RuntimeError(f"choices changed: {record['subject']}:{index}")
            if type(row["answer"]) is not int or not 0 <= row["answer"] <= 3:
                raise RuntimeError(f"answer changed: {record['subject']}:{index}")
        result[record["subject"]] = rows
    return result


def build_corpus(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    verify_sources(root, manifest)
    sources = load_rows(root, manifest)
    selected = []
    source_row = 0
    while len(selected) < 64:
        made_progress = False
        for subject in SUBJECTS:
            rows = sources[subject]
            if source_row >= len(rows):
                continue
            row = rows[source_row]
            selected.append(
                {
                    "choices": row["choices"],
                    "dataset": DATASET,
                    "dataset_revision": REVISION,
                    "expected_answer": "ABCD"[row["answer"]],
                    "item_index": len(selected),
                    "question": row["question"],
                    "source_row": source_row,
                    "subject": subject,
                }
            )
            made_progress = True
            if len(selected) == 64:
                break
        if not made_progress:
            raise RuntimeError("not enough source rows")
        source_row += 1
    return selected


def corpus_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(row) for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze or reproduce the pinned CP0-R1 MMLU-64 corpus"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_SOURCE_MANIFEST,
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--freeze-source-manifest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.freeze_source_manifest:
        if args.source_manifest.exists():
            raise RuntimeError("refusing to replace source manifest")
        args.source_manifest.write_bytes(canonical_bytes(build_source_manifest(args.source_root)))
    manifest = load_source_manifest(args.source_manifest)
    raw = corpus_bytes(build_corpus(args.source_root, manifest))
    if args.corpus.exists():
        if args.corpus.read_bytes() != raw:
            raise RuntimeError("frozen corpus differs")
    else:
        args.corpus.write_bytes(raw)
    print(f"{sha256_bytes(raw)}  {args.corpus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
