#!/usr/bin/env python3
"""Build the frozen S33 WikiText prompt-token corpus."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


SCHEMA = "s33-wikitext-prompt-v1"
MANIFEST_SCHEMA = "s33-wikitext-corpus-manifest-v1"
SOURCE_SHA256 = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"
SOURCE_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SOURCE_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/"
    f"{SOURCE_REVISION}/wikitext-2-raw-v1/test-00000-of-00001.parquet"
)
PROMPT_COUNT = 128
PROMPT_TOKENS = 8


class CorpusError(ValueError):
    pass


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def candidate_rows(values: list[object]) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for source_row, value in enumerate(values):
        if type(value) is not str:
            raise CorpusError("source text column contains a non-string value")
        text = value.strip()
        if not text or text.startswith("="):
            continue
        result.append((source_row, text))
    return result


def parse_token_ids(raw: bytes) -> list[int]:
    try:
        value = ast.literal_eval(raw.decode("ascii").strip())
    except (UnicodeError, SyntaxError, ValueError) as exc:
        raise CorpusError("tokenizer output is malformed") from exc
    if type(value) is not list or not value:
        raise CorpusError("tokenizer returned an empty or non-list value")
    if any(type(token) is not int or token < 0 for token in value):
        raise CorpusError("tokenizer returned an invalid token id")
    return value


def tokenize(tokenizer: Path, model: Path, text: str) -> list[int]:
    completed = subprocess.run(
        [
            str(tokenizer), "-m", str(model), "--ids", "--log-disable",
            "--stdin", "--no-escape",
        ],
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise CorpusError(f"tokenizer failed with exit {completed.returncode}")
    return parse_token_ids(completed.stdout)


def build_records(
    values: list[object], tokenizer: Path, model: Path, workers: int,
) -> list[dict[str, object]]:
    candidates = candidate_rows(values)
    if len(candidates) < PROMPT_COUNT:
        raise CorpusError("source has too few candidate rows")
    records: list[dict[str, object]] = []
    batch_size = min(len(candidates), PROMPT_COUNT * 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tokenized = list(pool.map(
            lambda item: tokenize(tokenizer, model, item[1]),
            candidates[:batch_size],
        ))
    for (source_row, text), tokens in zip(candidates[:batch_size], tokenized):
        if len(tokens) < PROMPT_TOKENS:
            continue
        records.append({
            "schema": SCHEMA,
            "prompt_id": len(records),
            "source_row": source_row,
            "text_sha256": sha256(text.encode("utf-8")),
            "source_token_count": len(tokens),
            "tokens": tokens[:PROMPT_TOKENS],
        })
        if len(records) == PROMPT_COUNT:
            return records
    raise CorpusError("candidate prefix did not yield 128 tokenizable prompts")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists() or args.manifest.exists()
        or not 1 <= args.workers <= 16
        or not args.source.is_file()
        or not args.tokenizer.is_file()
        or not args.model.is_file()
    ):
        parser.error("invalid corpus configuration")
    source_raw = args.source.read_bytes()
    if sha256(source_raw) != SOURCE_SHA256:
        parser.error("source digest mismatch")
    try:
        import pyarrow
        import pyarrow.parquet as parquet

        table = parquet.read_table(args.source, columns=["text"])
        values = table.column("text").to_pylist()
        records = build_records(values, args.tokenizer, args.model, args.workers)
    except (CorpusError, OSError, subprocess.SubprocessError) as exc:
        print(f"S33_CORPUS_FAIL: {exc}")
        return 2
    output_raw = b"".join(canonical(record) for record in records)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source_url": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "source_rows": len(values),
        "selection": {
            "order": "source_row_ascending",
            "exclude_empty": True,
            "exclude_heading_prefix": "=",
            "minimum_tokens": PROMPT_TOKENS,
            "prompt_count": PROMPT_COUNT,
            "prompt_tokens": PROMPT_TOKENS,
        },
        "tokenizer_sha256": sha256(args.tokenizer.read_bytes()),
        "model_sha256": sha256(args.model.read_bytes()),
        "pyarrow_version": pyarrow.__version__,
        "output_sha256": sha256(output_raw),
        "output_records": len(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output_raw)
    args.manifest.write_bytes(canonical(manifest))
    print(canonical(manifest).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
