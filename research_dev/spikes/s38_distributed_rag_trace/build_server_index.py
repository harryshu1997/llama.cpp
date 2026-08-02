#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

import numpy as np


INDEX_SCHEMA = "s38-bge-index-v1"
CLS_TOKEN = 101
SEP_TOKEN = 102


class IndexBuildError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def http_json(url: str, body: dict[str, Any] | None = None, timeout: float = 120.0) -> Any:
    data = None if body is None else canonical_json(body).encode("ascii")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, urllib.error.URLError) as error:
        raise IndexBuildError(f"request failed for {url}: {error}") from None


def load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        with path.open("r", encoding="ascii") as stream:
            for line_number, line in enumerate(stream, 1):
                value = json.loads(line)
                document_id = value.get("document_id")
                if not isinstance(document_id, str) or document_id in seen:
                    raise IndexBuildError(f"invalid document ID at {path}:{line_number}")
                seen.add(document_id)
                records.append(value)
    records.sort(key=lambda item: item["document_id"])
    if len(records) != 609:
        raise IndexBuildError(f"expected 609 documents, found {len(records)}")
    return records


def token_pieces(base_url: str, text: str) -> tuple[list[int], list[str]]:
    response = http_json(base_url + "/tokenize", {
        "content": text,
        "add_special": False,
        "parse_special": True,
        "with_pieces": True,
    })
    values = response.get("tokens") if isinstance(response, dict) else None
    if not isinstance(values, list):
        raise IndexBuildError("tokenize response has no token list")
    tokens: list[int] = []
    pieces: list[str] = []
    for value in values:
        if not isinstance(value, dict) or type(value.get("id")) is not int:
            raise IndexBuildError("tokenize response contains an invalid token")
        piece = value.get("piece")
        if isinstance(piece, list):
            if any(type(item) is not int or item < 0 or item > 255 for item in piece):
                raise IndexBuildError("tokenize response contains invalid piece bytes")
            piece = bytes(piece).decode("utf-8", errors="replace")
        if not isinstance(piece, str):
            raise IndexBuildError("tokenize response contains a non-string piece")
        tokens.append(value["id"])
        pieces.append(piece)
    return tokens, pieces


def make_chunks(
    documents: list[dict[str, Any]],
    base_url: str,
    chunk_tokens: int,
    overlap_tokens: int,
) -> tuple[list[dict[str, Any]], list[list[int]]]:
    stride = chunk_tokens - overlap_tokens
    if chunk_tokens < 32 or overlap_tokens < 0 or stride < 1:
        raise IndexBuildError("invalid chunk policy")
    chunks: list[dict[str, Any]] = []
    model_inputs: list[list[int]] = []
    for document in documents:
        title = document.get("title") or ""
        body = document.get("body") or ""
        if not isinstance(title, str) or not isinstance(body, str):
            raise IndexBuildError(f"document {document['document_id']} has invalid text")
        tokens, pieces = token_pieces(base_url, title + "\n\n" + body)
        if not tokens:
            continue
        chunk_index = 0
        for start in range(0, len(tokens), stride):
            end = min(start + chunk_tokens, len(tokens))
            content_tokens = tokens[start:end]
            text = "".join(pieces[start:end]).strip()
            if not text:
                raise IndexBuildError(f"empty chunk for {document['document_id']}")
            chunks.append({
                "schema_version": 1,
                "chunk_id": f"{document['document_id']}:chunk:{chunk_index}",
                "document_id": document["document_id"],
                "shard": document["shard"],
                "title": title,
                "url": document["url"],
                "token_start": start,
                "token_end": end,
                "token_count": len(content_tokens),
                "text": text,
            })
            model_inputs.append([CLS_TOKEN, *content_tokens, SEP_TOKEN])
            chunk_index += 1
            if end == len(tokens):
                break
    return chunks, model_inputs


def embed_batches(base_url: str, token_rows: list[list[int]], batch_size: int) -> np.ndarray:
    embeddings: list[list[float]] = []
    for start in range(0, len(token_rows), batch_size):
        inputs = token_rows[start:start + batch_size]
        response = http_json(base_url + "/v1/embeddings", {
            "input": inputs,
            "model": "bge-small-en-v1.5",
            "encoding_format": "float",
        }, timeout=300.0)
        data = response.get("data") if isinstance(response, dict) else None
        if not isinstance(data, list) or len(data) != len(inputs):
            raise IndexBuildError("embedding response count mismatch")
        ordered = sorted(data, key=lambda item: item.get("index", -1))
        for expected_index, item in enumerate(ordered):
            if item.get("index") != expected_index or not isinstance(item.get("embedding"), list):
                raise IndexBuildError("embedding response has invalid indexes")
            row = item["embedding"]
            if not row or any(type(value) not in (int, float) for value in row):
                raise IndexBuildError("embedding response contains invalid values")
            embeddings.append(row)
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(token_rows):
        raise IndexBuildError("embedding matrix has an invalid shape")
    if not np.isfinite(matrix).all():
        raise IndexBuildError("embedding matrix contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 0):
        raise IndexBuildError("embedding matrix contains a zero row")
    matrix /= norms[:, None]
    return matrix


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npy(path: Path, matrix: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, matrix, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the S38 all-server BGE index")
    parser.add_argument("--documents", type=Path, action="append", required=True)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:18081")
    parser.add_argument("--embedding-model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--overlap-tokens", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.batch_size < 1 or args.batch_size > 8:
        raise IndexBuildError("batch-size must be in [1, 8]")
    if len(args.embedding_model_sha256) != 64:
        raise IndexBuildError("embedding model SHA-256 is invalid")
    health = http_json(args.embedding_url + "/health")
    if health.get("status") != "ok":
        raise IndexBuildError("embedding server is not ready")

    documents = load_jsonl([path.resolve() for path in args.documents])
    chunks, token_rows = make_chunks(
        documents, args.embedding_url, args.chunk_tokens, args.overlap_tokens,
    )
    matrix = embed_batches(args.embedding_url, token_rows, args.batch_size)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    chunks_path = output / "chunks.jsonl"
    embeddings_path = output / "embeddings.npy"
    atomic_text(chunks_path, "".join(canonical_json(item) + "\n" for item in chunks))
    atomic_npy(embeddings_path, matrix)
    manifest = {
        "schema": INDEX_SCHEMA,
        "embedding_model_sha256": args.embedding_model_sha256,
        "source_documents": [
            {"path": path.name, "sha256": file_sha256(path)}
            for path in args.documents
        ],
        "chunk_policy": {
            "content_tokens": args.chunk_tokens,
            "overlap_tokens": args.overlap_tokens,
            "special_tokens": [CLS_TOKEN, SEP_TOKEN],
            "text_form": "concatenated tokenizer pieces with outer whitespace stripped",
        },
        "documents": len(documents),
        "chunks": len(chunks),
        "embedding_dimensions": int(matrix.shape[1]),
        "outputs": {
            "chunks": {"path": chunks_path.name, "sha256": file_sha256(chunks_path)},
            "embeddings": {"path": embeddings_path.name, "sha256": file_sha256(embeddings_path)},
        },
    }
    manifest_path = output / "manifest.json"
    atomic_text(manifest_path, canonical_json(manifest) + "\n")
    print(canonical_json({
        "status": "S38_INDEX_BUILT",
        "documents": len(documents),
        "chunks": len(chunks),
        "dimensions": int(matrix.shape[1]),
        "manifest_sha256": file_sha256(manifest_path),
    }))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (IndexBuildError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"S38_INDEX_ERROR: {error}") from None
