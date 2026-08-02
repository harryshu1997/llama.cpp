#!/usr/bin/env python3

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import threading
import time
from typing import Any
import urllib.error
import urllib.request

import numpy as np


RUN_SCHEMA = "s38-all-local-rag-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class RunError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RunError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def http_json(url: str, body: dict[str, Any] | None = None, timeout: float = 600.0) -> Any:
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
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RunError(f"HTTP {error.code} from {url}: {detail[:1000]}") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, urllib.error.URLError) as error:
        raise RunError(f"request failed for {url}: {error}") from None


def load_jsonl(path: Path, key: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    values: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="ascii") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line, object_pairs_hook=reject_duplicate_keys)
            record_key = value.get(key) if isinstance(value, dict) else None
            if not isinstance(record_key, str) or record_key in by_key:
                raise RunError(f"invalid {key} at {path}:{line_number}")
            values.append(value)
            by_key[record_key] = value
    return values, by_key


def load_index(index_dir: Path, expected_model_hash: str) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="ascii"),
        object_pairs_hook=reject_duplicate_keys,
    )
    if manifest.get("schema") != "s38-bge-index-v1":
        raise RunError("unexpected index schema")
    if manifest.get("embedding_model_sha256") != expected_model_hash:
        raise RunError("index embedding-model hash mismatch")
    chunks_path = index_dir / manifest["outputs"]["chunks"]["path"]
    embeddings_path = index_dir / manifest["outputs"]["embeddings"]["path"]
    if file_sha256(chunks_path) != manifest["outputs"]["chunks"]["sha256"]:
        raise RunError("chunk metadata hash mismatch")
    if file_sha256(embeddings_path) != manifest["outputs"]["embeddings"]["sha256"]:
        raise RunError("embedding matrix hash mismatch")
    chunks, _ = load_jsonl(chunks_path, "chunk_id")
    matrix = np.load(embeddings_path, allow_pickle=False)
    if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise RunError("embedding matrix shape or type mismatch")
    if matrix.shape[1] != manifest.get("embedding_dimensions") or not np.isfinite(matrix).all():
        raise RunError("embedding matrix content mismatch")
    return chunks, matrix, manifest


def endpoint_props(base_url: str) -> dict[str, Any]:
    health = http_json(base_url + "/health", timeout=5.0)
    if health.get("status") != "ok":
        raise RunError(f"endpoint is not ready: {base_url}")
    props = http_json(base_url + "/props", timeout=5.0)
    if not isinstance(props, dict) or not isinstance(props.get("model_path"), str):
        raise RunError(f"endpoint has invalid properties: {base_url}")
    return {
        "url": base_url,
        "model_path": props["model_path"],
        "model_alias": props.get("model_alias"),
        "total_slots": props.get("total_slots"),
        "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
    }


def verify_endpoint_model(label: str, props: dict[str, Any], expected_hash: str) -> None:
    model_path = Path(props["model_path"])
    if not model_path.is_file():
        raise RunError(f"{label} endpoint model is not a local file: {model_path}")
    actual_hash = file_sha256(model_path)
    if actual_hash != expected_hash:
        raise RunError(
            f"{label} endpoint model hash mismatch: expected {expected_hash}, got {actual_hash}"
        )


def query_embedding(base_url: str, query: str) -> np.ndarray:
    response = http_json(base_url + "/v1/embeddings", {
        "input": QUERY_PREFIX + query,
        "model": "bge-small-en-v1.5",
        "encoding_format": "float",
    })
    data = response.get("data") if isinstance(response, dict) else None
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0].get("embedding"), list):
        raise RunError("invalid query embedding response")
    embedding = np.asarray(data[0]["embedding"], dtype=np.float32)
    if embedding.ndim != 1 or not np.isfinite(embedding).all():
        raise RunError("invalid query embedding vector")
    norm = float(np.linalg.norm(embedding))
    if not math.isfinite(norm) or norm <= 0:
        raise RunError("query embedding has invalid norm")
    return embedding / norm


def retrieve(
    query: np.ndarray,
    chunks: list[dict[str, Any]],
    matrix: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    if query.shape[0] != matrix.shape[1]:
        raise RunError("query and index dimensions differ")
    scores = matrix @ query
    order = np.argsort(-scores, kind="stable")
    selected: list[dict[str, Any]] = []
    seen_documents: set[str] = set()
    for raw_index in order:
        index = int(raw_index)
        document_id = chunks[index]["document_id"]
        if document_id in seen_documents:
            continue
        seen_documents.add(document_id)
        selected.append({
            "chunk_index": index,
            "chunk_id": chunks[index]["chunk_id"],
            "document_id": document_id,
            "retrieval_score": float(scores[index]),
            "text": chunks[index]["text"],
            "title": chunks[index]["title"],
            "url": chunks[index]["url"],
            "shard": chunks[index]["shard"],
        })
        if len(selected) == top_k:
            break
    if len(selected) != top_k:
        raise RunError("index does not contain enough distinct documents")
    return selected


def rerank(base_url: str, query: str, candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    response = http_json(base_url + "/rerank", {
        "query": query,
        "documents": [item["text"] for item in candidates],
        "top_n": top_n,
    })
    results = response.get("results") if isinstance(response, dict) else None
    if not isinstance(results, list) or len(results) != top_n:
        raise RunError("invalid reranker response count")
    for result in results:
        if not isinstance(result, dict):
            raise RunError("reranker returned a non-object result")
        index = result.get("index")
        score = result.get("relevance_score")
        if type(index) is not int or index < 0 or index >= len(candidates):
            raise RunError("reranker returned an invalid index")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise RunError("reranker returned an invalid score")
    ranked: list[dict[str, Any]] = []
    seen: set[int] = set()
    for result in sorted(results, key=lambda item: (-item.get("relevance_score", float("-inf")), item.get("index", -1))):
        index = result.get("index")
        score = result.get("relevance_score")
        if type(index) is not int or index < 0 or index >= len(candidates) or index in seen:
            raise RunError("reranker returned an invalid index")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise RunError("reranker returned an invalid score")
        item = dict(candidates[index])
        item["rerank_score"] = float(score)
        ranked.append(item)
        seen.add(index)
    return ranked


def build_messages(query: str, ranked: list[dict[str, Any]]) -> list[dict[str, str]]:
    sources: list[str] = []
    for index, item in enumerate(ranked, 1):
        sources.append(
            f"[Source {index}] {item['title']}\nURL: {item['url']}\n{item['text']}"
        )
    return [
        {
            "role": "system",
            "content": "Answer using only the supplied sources. Return only the shortest final answer, with no explanation, reasoning, or citations.",
        },
        {
            "role": "user",
            "content": f"Question: {query}\n\nSources:\n\n" + "\n\n".join(sources),
        },
    ]


def generate(
    base_url: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    seed: int,
    reasoning_budget: int,
) -> tuple[str, str, dict[str, Any], str | None]:
    response = http_json(base_url + "/v1/chat/completions", {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "stream": False,
        "cache_prompt": False,
        "thinking_budget_tokens": reasoning_budget,
    }, timeout=1800.0)
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise RunError("invalid generation response")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise RunError("generation response has no text")
    reasoning_content = message.get("reasoning_content", "")
    if not isinstance(reasoning_content, str):
        raise RunError("generation response has invalid reasoning text")
    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        raise RunError("generation response has invalid usage")
    finish_reason = choices[0].get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise RunError("generation response has invalid finish reason")
    return content, reasoning_content, usage, finish_reason


def normalize_answer(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def answer_scores(prediction: str, reference: str) -> tuple[float, float]:
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    exact = float(pred == ref)
    if not pred or not ref:
        return exact, float(pred == ref)
    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    common = sum(min(count, ref_counts.get(token, 0)) for token, count in pred_counts.items())
    if common == 0:
        return exact, 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return exact, 2 * precision * recall / (precision + recall)


def recall(found: list[str], expected: list[str]) -> float | None:
    if not expected:
        return None
    return len(set(found) & set(expected)) / len(set(expected))


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise RunError("cannot compute a percentile of an empty list")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an all-local S38 RAG baseline")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:18081")
    parser.add_argument("--reranker-url", default="http://127.0.0.1:18082")
    parser.add_argument("--generation-url", default="http://127.0.0.1:18083")
    parser.add_argument("--embedding-model-sha256", required=True)
    parser.add_argument("--reranker-model-sha256", required=True)
    parser.add_argument("--generation-model-sha256", required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--rerank-top-n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reasoning-budget", type=int, default=64)
    parser.add_argument("--arrival-mode", choices=("trace", "burst"), default="trace")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.workers < 1 or args.workers > 128:
        raise RunError("workers must be in [1, 128]")
    if not 1 <= args.rerank_top_n <= args.retrieval_top_k:
        raise RunError("invalid retrieval/rerank depths")
    if args.reasoning_budget < 0:
        raise RunError("reasoning-budget must be nonnegative")
    for digest in (args.embedding_model_sha256, args.reranker_model_sha256, args.generation_model_sha256):
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RunError("invalid model SHA-256")
    output = args.output.resolve()
    if output.exists():
        raise RunError(f"output path already exists: {output}")

    requests, _ = load_jsonl(args.trace.resolve(), "event_id")
    _, payload_by_id = load_jsonl(args.payloads.resolve(), "payload_id")
    chunks, matrix, index_manifest = load_index(args.index.resolve(), args.embedding_model_sha256)
    endpoint_info = {
        "embedding": endpoint_props(args.embedding_url),
        "reranker": endpoint_props(args.reranker_url),
        "generation": endpoint_props(args.generation_url),
    }
    verify_endpoint_model("embedding", endpoint_info["embedding"], args.embedding_model_sha256)
    verify_endpoint_model("reranker", endpoint_info["reranker"], args.reranker_model_sha256)
    verify_endpoint_model("generation", endpoint_info["generation"], args.generation_model_sha256)
    if not requests:
        raise RunError("trace is empty")
    previous_time = -1
    for request in requests:
        if type(request.get("t_us")) is not int or request["t_us"] < previous_time:
            raise RunError("trace timestamps are not nondecreasing integers")
        previous_time = request["t_us"]
        payload_id = request.get("source_fields", {}).get("payload_id")
        if payload_id not in payload_by_id:
            raise RunError(f"missing payload for {request.get('event_id')}")

    origin_ns = time.monotonic_ns()
    lock = threading.Lock()
    completed = 0

    def run_one(request: dict[str, Any]) -> dict[str, Any]:
        nonlocal completed
        start_ns = time.monotonic_ns()
        payload = payload_by_id[request["source_fields"]["payload_id"]]
        query = payload["query"]

        stage_ns = start_ns
        embedding = query_embedding(args.embedding_url, query)
        after_embed_ns = time.monotonic_ns()
        candidates = retrieve(embedding, chunks, matrix, args.retrieval_top_k)
        after_retrieve_ns = time.monotonic_ns()
        ranked = rerank(args.reranker_url, query, candidates, args.rerank_top_n)
        after_rerank_ns = time.monotonic_ns()
        reasoning_budget = min(args.reasoning_budget, request["output_tokens"] // 2)
        answer, reasoning_content, usage, finish_reason = generate(
            args.generation_url,
            build_messages(query, ranked),
            request["output_tokens"],
            args.seed,
            reasoning_budget,
        )
        end_ns = time.monotonic_ns()

        evidence = payload["evidence_document_ids"]
        retrieved_ids = [item["document_id"] for item in candidates]
        reranked_ids = [item["document_id"] for item in ranked]
        exact, f1 = answer_scores(answer, payload["answer"])
        due_us = 0 if args.arrival_mode == "burst" else request["t_us"]
        worker_start_us = (start_ns - origin_ns) // 1000
        finish_us = (end_ns - origin_ns) // 1000
        result = {
            "schema": RUN_SCHEMA,
            "event_id": request["event_id"],
            "payload_id": payload["payload_id"],
            "arrival_due_us": due_us,
            "worker_start_us": worker_start_us,
            "finish_us": finish_us,
            "client_queue_us": max(0, worker_start_us - due_us),
            "service_us": (end_ns - start_ns) // 1000,
            "response_us": max(0, finish_us - due_us),
            "stages_us": {
                "query_embedding": (after_embed_ns - stage_ns) // 1000,
                "retrieval": (after_retrieve_ns - after_embed_ns) // 1000,
                "reranking": (after_rerank_ns - after_retrieve_ns) // 1000,
                "generation": (end_ns - after_rerank_ns) // 1000,
            },
            "requested_input_tokens": request["input_tokens"],
            "requested_output_tokens": request["output_tokens"],
            "realized_prompt_tokens": usage.get("prompt_tokens"),
            "realized_output_tokens": usage.get("completion_tokens"),
            "generation_finish_reason": finish_reason,
            "reasoning_budget_tokens": reasoning_budget,
            "reasoning_content": reasoning_content,
            "retrieved_chunk_ids": [item["chunk_id"] for item in candidates],
            "reranked_chunk_ids": [item["chunk_id"] for item in ranked],
            "retrieval_evidence_recall": recall(retrieved_ids, evidence),
            "rerank_evidence_recall": recall(reranked_ids, evidence),
            "answer_exact_match": exact,
            "answer_token_f1": f1,
            "reference_answer": payload["answer"],
            "generated_answer": answer,
        }
        with lock:
            completed += 1
            print(canonical_json({
                "completed": completed,
                "total": len(requests),
                "event_id": request["event_id"],
                "service_ms": round(result["service_us"] / 1000, 3),
                "response_ms": round(result["response_us"] / 1000, 3),
            }), flush=True)
        return result

    futures: list[concurrent.futures.Future[dict[str, Any]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for request in requests:
            due_us = 0 if args.arrival_mode == "burst" else request["t_us"]
            while True:
                remaining_ns = origin_ns + due_us * 1000 - time.monotonic_ns()
                if remaining_ns <= 0:
                    break
                time.sleep(min(remaining_ns / 1e9, 0.01))
            futures.append(executor.submit(run_one, request))
        results = [future.result() for future in futures]
    finish_ns = time.monotonic_ns()
    results.sort(key=lambda item: (item["arrival_due_us"], item["event_id"]))

    evidence_results = [item for item in results if item["retrieval_evidence_recall"] is not None]
    realized_prompt = [item["realized_prompt_tokens"] for item in results]
    realized_output = [item["realized_output_tokens"] for item in results]
    if any(type(value) is not int or value < 0 for value in realized_prompt):
        raise RunError("generation usage lacks valid prompt-token counts")
    if any(type(value) is not int or value < 0 for value in realized_output):
        raise RunError("generation usage lacks valid completion-token counts")
    wall_s = (finish_ns - origin_ns) / 1e9
    summary = {
        "schema": RUN_SCHEMA,
        "status": "S38_ALL_LOCAL_BASELINE_COMPLETE",
        "host_label": args.host_label,
        "requests": len(results),
        "arrival_mode": args.arrival_mode,
        "workers": args.workers,
        "retrieval_top_k": args.retrieval_top_k,
        "rerank_top_n": args.rerank_top_n,
        "wall_s": wall_s,
        "requests_per_s": len(results) / wall_s,
        "requested_input_tokens": sum(item["requested_input_tokens"] for item in results),
        "requested_output_tokens": sum(item["requested_output_tokens"] for item in results),
        "realized_prompt_tokens": sum(realized_prompt),
        "realized_output_tokens": sum(realized_output),
        "realized_output_tokens_per_s": sum(realized_output) / wall_s,
        "client_queue_ms": {
            "p50": percentile([item["client_queue_us"] / 1000 for item in results], 0.50),
            "p95": percentile([item["client_queue_us"] / 1000 for item in results], 0.95),
            "max": max(item["client_queue_us"] / 1000 for item in results),
        },
        "service_ms": {
            "p50": percentile([item["service_us"] / 1000 for item in results], 0.50),
            "p95": percentile([item["service_us"] / 1000 for item in results], 0.95),
            "max": max(item["service_us"] / 1000 for item in results),
        },
        "response_ms": {
            "p50": percentile([item["response_us"] / 1000 for item in results], 0.50),
            "p95": percentile([item["response_us"] / 1000 for item in results], 0.95),
            "max": max(item["response_us"] / 1000 for item in results),
        },
        "stage_p50_ms": {
            stage: statistics.median(item["stages_us"][stage] / 1000 for item in results)
            for stage in ("query_embedding", "retrieval", "reranking", "generation")
        },
        "quality": {
            "evidence_queries": len(evidence_results),
            "retrieval_mean_recall": statistics.mean(item["retrieval_evidence_recall"] for item in evidence_results),
            "retrieval_complete_rate": statistics.mean(item["retrieval_evidence_recall"] == 1.0 for item in evidence_results),
            "rerank_mean_recall": statistics.mean(item["rerank_evidence_recall"] for item in evidence_results),
            "rerank_complete_rate": statistics.mean(item["rerank_evidence_recall"] == 1.0 for item in evidence_results),
            "answer_exact_match": statistics.mean(item["answer_exact_match"] for item in results),
            "answer_token_f1": statistics.mean(item["answer_token_f1"] for item in results),
        },
        "models": {
            "embedding_sha256": args.embedding_model_sha256,
            "reranker_sha256": args.reranker_model_sha256,
            "generation_sha256": args.generation_model_sha256,
        },
        "endpoints": endpoint_info,
        "inputs": {
            "trace_sha256": file_sha256(args.trace.resolve()),
            "payloads_sha256": file_sha256(args.payloads.resolve()),
            "index_manifest_sha256": file_sha256(args.index.resolve() / "manifest.json"),
            "index_chunks": index_manifest["chunks"],
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    requests_path = output / "requests.jsonl"
    summary_path = output / "summary.json"
    atomic_text(requests_path, "".join(canonical_json(item) + "\n" for item in results))
    atomic_text(summary_path, canonical_json(summary) + "\n")
    manifest = {
        "schema": RUN_SCHEMA,
        "requests": {"path": requests_path.name, "sha256": file_sha256(requests_path)},
        "summary": {"path": summary_path.name, "sha256": file_sha256(summary_path)},
    }
    atomic_text(output / "manifest.json", canonical_json(manifest) + "\n")
    print(canonical_json(summary), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RunError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"S38_RUN_ERROR: {error}") from None
