#!/usr/bin/env python3

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
from typing import Any


VERSION = "s38-distributed-rag-trace-v1"

MULTIHOP_QUERY_SHA256 = "03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff"
MULTIHOP_CORPUS_SHA256 = "20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f"
RAGPULSE_SHA256 = "cd371571bef3320907147f8901729e37f412aafb067ec8eb153e2828e1801e65"

HASH_ID_ORDER = ("sys_prompt", "passages_ids", "history", "web_search", "user_input")
PHONE_NAMES = ("op12", "op15")


class BuildError(RuntimeError):
    pass


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BuildError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)


def load_ragpulse(path: Path) -> list[tuple[int, dict[str, Any]]]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[tuple[int, dict[str, Any]]] = []
    blank_lines: list[int] = []
    for line_number, line in enumerate(raw_lines, 1):
        if not line:
            blank_lines.append(line_number)
            continue
        value = json.loads(line, object_pairs_hook=reject_duplicate_keys)
        if not isinstance(value, dict):
            raise BuildError(f"RAGPulse line {line_number} is not an object")
        rows.append((line_number, value))

    if blank_lines != [len(raw_lines)]:
        raise BuildError(f"unexpected RAGPulse blank lines: {blank_lines}")
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise BuildError(f"hash mismatch for {path}: expected {expected}, got {actual}")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        for value in values:
            stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def document_id(url: str) -> str:
    return "doc:" + hashlib.sha256(url.encode("utf-8")).hexdigest()


def document_shard(doc_id: str) -> str:
    value = int(doc_id.removeprefix("doc:"), 16)
    return PHONE_NAMES[value % len(PHONE_NAMES)]


def validate_sources(
    queries: Any,
    corpus: Any,
    ragpulse: list[tuple[int, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(queries, list) or len(queries) != 2556:
        raise BuildError("MultiHop-RAG must contain exactly 2556 queries")
    if not isinstance(corpus, list) or len(corpus) != 609:
        raise BuildError("MultiHop-RAG corpus must contain exactly 609 documents")
    if len(ragpulse) != 7106:
        raise BuildError("RAGPulse must contain exactly 7106 records")

    documents_by_url: dict[str, dict[str, Any]] = {}
    for index, document in enumerate(corpus):
        if not isinstance(document, dict):
            raise BuildError(f"corpus row {index} is not an object")
        url = document.get("url")
        if not isinstance(url, str) or not url:
            raise BuildError(f"corpus row {index} has no URL")
        if url in documents_by_url:
            raise BuildError(f"duplicate corpus URL: {url}")
        documents_by_url[url] = document

    for index, query in enumerate(queries):
        if not isinstance(query, dict):
            raise BuildError(f"query row {index} is not an object")
        for key in ("query", "answer", "question_type", "evidence_list"):
            if key not in query:
                raise BuildError(f"query row {index} lacks {key}")
        evidence = query["evidence_list"]
        if not isinstance(evidence, list):
            raise BuildError(f"query row {index} evidence is not a list")
        for item in evidence:
            if not isinstance(item, dict) or item.get("url") not in documents_by_url:
                raise BuildError(f"query row {index} references an unknown document")

    required_hash_keys = set(HASH_ID_ORDER)
    for line_number, row in ragpulse:
        for key in ("timestamp", "input_length", "output_length", "hash_ids", "session_id"):
            if key not in row:
                raise BuildError(f"RAGPulse line {line_number} lacks {key}")
        timestamp = row["timestamp"]
        if not isinstance(timestamp, str) or not timestamp.isascii() or not timestamp.isdigit():
            raise BuildError(f"RAGPulse line {line_number} has an invalid timestamp")
        if type(row["input_length"]) is not int or type(row["output_length"]) is not int:
            raise BuildError(f"RAGPulse line {line_number} has a non-integer token count")
        hash_ids = row["hash_ids"]
        if not isinstance(hash_ids, dict) or set(hash_ids) != required_hash_keys:
            raise BuildError(f"RAGPulse line {line_number} has an invalid hash_ids object")
        if any(not isinstance(hash_ids[key], list) for key in HASH_ID_ORDER):
            raise BuildError(f"RAGPulse line {line_number} has invalid hash ID lists")

    return queries, corpus


def build_documents(corpus: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    documents: list[dict[str, Any]] = []
    id_by_url: dict[str, str] = {}
    for source_index, source in enumerate(corpus):
        url = source["url"]
        doc_id = document_id(url)
        shard = document_shard(doc_id)
        id_by_url[url] = doc_id
        documents.append({
            "schema_version": 1,
            "document_id": doc_id,
            "source_index": source_index,
            "shard": shard,
            "title": source.get("title"),
            "author": source.get("author"),
            "source": source.get("source"),
            "category": source.get("category"),
            "published_at": source.get("published_at"),
            "url": url,
            "body": source.get("body"),
        })
    documents.sort(key=lambda item: item["document_id"])
    return documents, id_by_url


def query_order(count: int) -> list[int]:
    salt = VERSION + ":query-order:"
    return sorted(range(count), key=lambda index: hashlib.sha256(f"{salt}{index}".encode("ascii")).digest())


def cache_keys(hash_ids: dict[str, list[int]]) -> list[str]:
    result: list[str] = []
    for namespace in HASH_ID_ORDER:
        result.extend(f"{namespace}:{value}" for value in hash_ids[namespace])
    return result


def build_payloads(
    queries: list[dict[str, Any]],
    id_by_url: dict[str, str],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for query_index, source in enumerate(queries):
        evidence_ids = [id_by_url[item["url"]] for item in source["evidence_list"]]
        shards = sorted({document_shard(doc_id) for doc_id in evidence_ids})
        payloads.append({
            "schema_version": 1,
            "payload_id": f"multihoprag:{query_index}",
            "query_index": query_index,
            "query": source["query"],
            "answer": source["answer"],
            "question_type": source["question_type"],
            "evidence_document_ids": evidence_ids,
            "evidence_shards": shards,
        })
    return payloads


def build_requests(
    ragpulse: list[tuple[int, dict[str, Any]]],
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered_arrivals = sorted(ragpulse, key=lambda item: (int(item[1]["timestamp"]), item[0]))
    selected_arrivals = ordered_arrivals[:len(payloads)]
    selected_payloads = [payloads[index] for index in query_order(len(payloads))]
    t0_s = int(selected_arrivals[0][1]["timestamp"])

    requests: list[dict[str, Any]] = []
    for (line_number, arrival), payload in zip(selected_arrivals, selected_payloads, strict=True):
        hash_ids = arrival["hash_ids"]
        requests.append({
            "schema_version": 1,
            "event_id": f"ragpulse:{line_number}:multihoprag:{payload['query_index']}",
            "source": "ragpulse-multihoprag-v1",
            "provenance": "semi_synthetic",
            "t_us": (int(arrival["timestamp"]) - t0_s) * 1_000_000,
            "service": "distributed_rag_qa",
            "model_class": "gemma4_12b_distributed_rag",
            "session_id": arrival["session_id"],
            "input_tokens": arrival["input_length"],
            "output_tokens": arrival["output_length"],
            "images": 0,
            "audio_ms": 0,
            "retrieved_chunks": len(hash_ids["passages_ids"]),
            "cache_keys": cache_keys(hash_ids),
            "observed_latency_us": None,
            "priority_class": None,
            "deadline_us": None,
            "priority_provenance": "none",
            "deadline_provenance": "none",
            "source_fields": {
                "ragpulse_source_line": line_number,
                "ragpulse_timestamp_s": int(arrival["timestamp"]),
                "payload_id": payload["payload_id"],
                "question_type": payload["question_type"],
                "evidence_count": len(payload["evidence_document_ids"]),
                "evidence_shards": ",".join(payload["evidence_shards"]),
            },
        })
    return requests


def dense_window(requests: list[dict[str, Any]], count: int, time_scale_den: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if count < 1 or count > len(requests):
        raise BuildError("dense count is outside the request count")
    if time_scale_den < 1:
        raise BuildError("time scale denominator must be positive")

    span, start = min(
        (requests[index + count - 1]["t_us"] - requests[index]["t_us"], index)
        for index in range(len(requests) - count + 1)
    )
    base = requests[start]["t_us"]
    dense: list[dict[str, Any]] = []
    for source in requests[start:start + count]:
        item = dict(source)
        item["t_us"] = (source["t_us"] - base) // time_scale_den
        dense.append(item)
    return dense, {
        "source_start_index": start,
        "source_end_index": start + count - 1,
        "source_span_us": span,
        "scaled_span_us": dense[-1]["t_us"],
        "time_scale_num": 1,
        "time_scale_den": time_scale_den,
    }


def output_record(path: Path, count: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "records": count,
        "sha256": file_sha256(path),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    query_path = args.multihop_queries.resolve()
    corpus_path = args.multihop_corpus.resolve()
    ragpulse_path = args.ragpulse.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    require_sha256(query_path, MULTIHOP_QUERY_SHA256)
    require_sha256(corpus_path, MULTIHOP_CORPUS_SHA256)
    require_sha256(ragpulse_path, RAGPULSE_SHA256)

    ragpulse = load_ragpulse(ragpulse_path)
    queries, corpus = validate_sources(
        load_json(query_path),
        load_json(corpus_path),
        ragpulse,
    )
    documents, id_by_url = build_documents(corpus)
    payloads = build_payloads(queries, id_by_url)
    requests = build_requests(ragpulse, payloads)
    dense, dense_meta = dense_window(requests, args.dense_count, args.time_scale_den)

    op12_documents = [item for item in documents if item["shard"] == "op12"]
    op15_documents = [item for item in documents if item["shard"] == "op15"]
    output_paths = {
        "op12_documents": output_dir / "op12_documents.jsonl",
        "op15_documents": output_dir / "op15_documents.jsonl",
        "payloads": output_dir / "payloads.jsonl",
        "requests": output_dir / "requests.jsonl",
        "dense_requests": output_dir / f"dense{args.dense_count}_x{args.time_scale_den}.jsonl",
    }
    write_jsonl(output_paths["op12_documents"], op12_documents)
    write_jsonl(output_paths["op15_documents"], op15_documents)
    write_jsonl(output_paths["payloads"], payloads)
    write_jsonl(output_paths["requests"], requests)
    write_jsonl(output_paths["dense_requests"], dense)

    question_types = collections.Counter(item["question_type"] for item in payloads)
    evidence_counts = collections.Counter(len(item["evidence_document_ids"]) for item in payloads)
    shard_patterns = collections.Counter(
        ",".join(item["evidence_shards"]) if item["evidence_shards"] else "none"
        for item in payloads
    )
    manifest = {
        "schema_version": 1,
        "builder_version": VERSION,
        "provenance": "semi_synthetic",
        "sources": {
            "multihop_queries": {
                "revision": "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82",
                "license": "ODC-BY-1.0",
                "sha256": MULTIHOP_QUERY_SHA256,
                "records": len(queries),
            },
            "multihop_corpus": {
                "revision": "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82",
                "license": "ODC-BY-1.0",
                "sha256": MULTIHOP_CORPUS_SHA256,
                "records": len(corpus),
            },
            "ragpulse": {
                "revision": "99a62769a91d5ebd17a2d4ddbbc88c1d16edc0e8",
                "license": "MIT",
                "sha256": RAGPULSE_SHA256,
                "records": len(ragpulse),
            },
        },
        "transforms": {
            "arrival_order": "stable sort by (integer timestamp, source line)",
            "arrival_selection": "first 2556 sorted RAGPulse records",
            "query_order": "ascending SHA-256 of builder-version/query-index",
            "document_id": "SHA-256 of UTF-8 URL",
            "document_shard": "integer document digest modulo 2; 0=op12, 1=op15",
            "dense_window": dense_meta,
            "priority_and_deadline": "unset; no real labels exist",
        },
        "statistics": {
            "documents": len(documents),
            "op12_documents": len(op12_documents),
            "op15_documents": len(op15_documents),
            "requests": len(requests),
            "question_types": dict(sorted(question_types.items())),
            "evidence_counts": {str(key): value for key, value in sorted(evidence_counts.items())},
            "evidence_shard_patterns": dict(sorted(shard_patterns.items())),
        },
        "outputs": {
            key: output_record(path, {
                "op12_documents": len(op12_documents),
                "op15_documents": len(op15_documents),
                "payloads": len(payloads),
                "requests": len(requests),
                "dense_requests": len(dense),
            }[key])
            for key, path in output_paths.items()
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the S38 distributed-RAG trace bundle")
    parser.add_argument("--multihop-queries", required=True, type=Path)
    parser.add_argument("--multihop-corpus", required=True, type=Path)
    parser.add_argument("--ragpulse", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dense-count", type=int, default=128)
    parser.add_argument("--time-scale-den", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        result = build(parse_args())
    except (BuildError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"S38_BUILD_ERROR: {error}") from None
    print(json.dumps(result["statistics"], sort_keys=True))
