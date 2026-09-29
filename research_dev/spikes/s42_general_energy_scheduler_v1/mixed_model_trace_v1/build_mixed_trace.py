#!/usr/bin/env python3
"""Build the deterministic six-model BurstGPT mixed trace."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
SOURCE = (
    REPO_ROOT
    / "research_dev/spikes/s41_gemma_qwen_continuous_baseline"
    / "tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1"
    / "REQUESTS_SEMANTIC_SOURCE.jsonl"
)
OUTPUT = HERE / "REQUESTS_MIXED_114.jsonl"
MANIFEST = HERE / "TRACE_MANIFEST.json"

SCHEMA = "s42-six-model-burstgpt-mixed-v1"
MANIFEST_SCHEMA = "s42-six-model-burstgpt-mixed-manifest-v1"
SOURCE_SHA256 = (
    "b20a9ba66ee3558d835a0e19ed3cfa4c"
    "31a4a9e8b4f9c085b29a14f80250a0ff"
)
SOURCE_SCHEMA = "s41-gemma-qwen-request-semantic-source-v1"
DONOR_INDICES = (0, 8, 16, 24, 32, 41, 49, 57, 65, 73)

MODEL_INVENTORY = {
    "gemma-4-12b-it-q4_0": {
        "artifact_bytes": 6_975_878_176,
        "artifact_file": "gemma-4-12B-it-Q4_0.gguf",
        "artifact_sha256": (
            "494518c2262a26e2a607af0e40bca11c"
            "4de5a0b108e7c21308906dbbfb1c6f8c"
        ),
        "kind": "text_decoder",
    },
    "gemma-4-e2b-it-q8_0-vlm": {
        "artifact_bytes": 4_967_495_040,
        "artifact_file": "gemma-4-E2B-it-Q8_0.gguf",
        "artifact_sha256": (
            "bc88cdd8470bd0b8a610f6fed4050b24"
            "67e1d28dfc546284630a0213a10436e9"
        ),
        "kind": "vision_language",
        "projector_bytes": 557_367_776,
        "projector_file": "mmproj-gemma-4-E2B-it-Q8_0.gguf",
        "projector_sha256": (
            "8a82e0fd831bb7cb5c8898b86393eb14"
            "042986b950a60e1034bf21d061aac8a8"
        ),
    },
    "llama-3.2-1b-instruct-q4_0": {
        "artifact_bytes": 770_928_288,
        "artifact_file": "Llama-3.2-1B-Instruct-Q4_0.gguf",
        "artifact_sha256": (
            "4b90b1d7ae7324676194755a6dfce11c"
            "b6e457982c4c01a1db2857be1ed064ad"
        ),
        "kind": "text_decoder",
    },
    "qwen3-0.6b-q8_0": {
        "artifact_bytes": 804_753_632,
        "artifact_file": "Qwen3-0.6B-Q8_0.gguf",
        "artifact_sha256": (
            "361cc68159042c36ebff7715dc5a2e46"
            "12153e88f3e9c9c234820849d6dc9e1d"
        ),
        "kind": "text_decoder",
    },
    "qwen3-8b-q8_0": {
        "artifact_bytes": 8_709_518_112,
        "artifact_file": "Qwen3-8B-Q8_0.gguf",
        "artifact_sha256": (
            "408b955510e196121c1c375201744783"
            "b5c9a43c7956d73fc78df54c66e883d6"
        ),
        "kind": "text_decoder",
    },
    "qwen3-14b-q4_k_m": {
        "artifact_bytes": 9_001_752_960,
        "artifact_file": "Qwen3-14B-Q4_K_M.gguf",
        "artifact_sha256": (
            "500a8806e85ee9c83f3ae084202955924"
            "51379b4f8cf2d0f41c15dffeb6b81f0"
        ),
        "kind": "text_decoder",
    },
}

TEXT_STREAMS = (
    {
        "arrival_offset_us": 100_000,
        "model_id": "qwen3-0.6b-q8_0",
        "stream_id": "qwen3-0.6b-overlay",
        "template": "qwen3",
        "tokenizer_model_id": "qwen3-0.6b-q8_0",
    },
    {
        "arrival_offset_us": 200_000,
        "model_id": "llama-3.2-1b-instruct-q4_0",
        "stream_id": "llama-3.2-1b-overlay",
        "template": "llama3",
        "tokenizer_model_id": "llama-3.2-1b-instruct-q4_0",
    },
    {
        "arrival_offset_us": 300_000,
        "model_id": "qwen3-8b-q8_0",
        "stream_id": "qwen3-8b-overlay",
        "template": "qwen3",
        "tokenizer_model_id": "qwen3-0.6b-q8_0",
    },
)

IMAGE_INVENTORY = {
    "moon-newspaper": {
        "bytes": 124_071,
        "height": 488,
        "media_type": "image/jpeg",
        "path": "tools/mtmd/test-1.jpeg",
        "sha256": (
            "2dff664c0c8aaea18aff8cbe7e868845"
            "b775e90cdd7a0bac98df709b131deaa3"
        ),
        "width": 640,
    },
    "lotus-flower": {
        "bytes": 817_630,
        "height": 3840,
        "media_type": "image/webp",
        "path": (
            "tools/ui/tests/stories/fixtures/assets/"
            "beautiful-flowers-lotus.webp"
        ),
        "sha256": (
            "a5f728fbce7c5b2ed8ef32303942f97"
            "a00e4cadb791ec65eadcfbe59ebe9625f"
        ),
        "width": 3840,
    },
    "android-studio": {
        "bytes": 490_930,
        "height": 2499,
        "media_type": "image/jpeg",
        "path": "docs/android/imported-into-android-studio.jpg",
        "sha256": (
            "3cc2c2bf6582bf748216ba5d19fc6ff3"
            "55d054f8f9d3321212407ace06e4b8d9"
        ),
        "width": 2939,
    },
}

VLM_CASES = (
    (
        "moon-newspaper",
        "What is the main headline?",
        (("men walk on moon",),),
    ),
    (
        "lotus-flower",
        "What type of flower is shown?",
        (("lotus", "water lily"),),
    ),
    (
        "android-studio",
        "Which development environment is shown?",
        (("android studio",),),
    ),
    (
        "moon-newspaper",
        "Which newspaper is shown?",
        (("new york times",),),
    ),
    (
        "lotus-flower",
        "What is the dominant petal color?",
        (("pink",),),
    ),
    (
        "android-studio",
        "Which C++ source file is open in the editor?",
        (("ai_chat.cpp", "ai chat.cpp"),),
    ),
    (
        "moon-newspaper",
        "What date is printed at the top?",
        (("july 21 1969",),),
    ),
    (
        "lotus-flower",
        "What color is the flower center?",
        (("yellow", "orange"),),
    ),
    (
        "android-studio",
        "What final build result is visible?",
        (("build successful",),),
    ),
    (
        "moon-newspaper",
        "What did the astronauts collect?",
        (("rocks",),),
    ),
)


def canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def check_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"file size mismatch: {path}")
    if digest_file(path) != expected_sha256:
        raise RuntimeError(f"file hash mismatch: {path}")


class Codec:
    def __init__(
        self,
        executable: Path,
        model: Path,
        model_sha256: str,
        library_dir: Path,
    ):
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = (
            f"{library_dir}:" + environment.get("LD_LIBRARY_PATH", "")
        )
        self.model_sha256 = model_sha256
        self.request_id = 0
        self.process = subprocess.Popen(
            [
                str(executable),
                "--model",
                str(model),
                "--model-sha256",
                model_sha256,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            bufsize=1,
        )

    def exchange(self, op: str, key: str, value: Any) -> dict[str, Any]:
        self.request_id += 1
        request = {
            key: value,
            "op": op,
            "request_id": self.request_id,
            "schema": "layersplit-token-codec-request-v1",
        }
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(canonical(request).decode("ascii"))
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("token codec response EOF")
        response = json.loads(line)
        if (
            response.get("schema") != "layersplit-token-codec-response-v1"
            or response.get("request_id") != self.request_id
            or response.get("op") != op
            or response.get("model_sha256") != self.model_sha256
        ):
            raise RuntimeError("token codec response identity")
        return response

    def tokenize(self, text: str) -> list[int]:
        response = self.exchange("tokenize", "text", text)
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise RuntimeError("token codec tokenize response")
        if any(type(token) is not int or token < 0 for token in tokens):
            raise RuntimeError("token codec token value")
        return tokens

    def detokenize(self, tokens: list[int]) -> str:
        response = self.exchange("detokenize", "tokens", tokens)
        text = response.get("text")
        if not isinstance(text, str):
            raise RuntimeError("token codec detokenize response")
        text.encode("ascii")
        return text

    def close(self) -> None:
        assert self.process.stdin is not None
        assert self.process.stderr is not None
        self.process.stdin.close()
        error = self.process.stderr.read()
        status = self.process.wait(timeout=30)
        if status != 0 or error:
            raise RuntimeError(
                f"token codec failed with status {status}: {error.strip()}"
            )


def chat_parts(template: str) -> tuple[str, str, int | None]:
    if template == "qwen3":
        return (
            "<|im_start|>user\n",
            "\n<|im_end|>\n<|im_start|>assistant\n",
            None,
        )
    if template == "llama3":
        return (
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            128000,
        )
    if template == "gemma4":
        return (
            "<|turn>user\n",
            "\n<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
            2,
        )
    raise RuntimeError(f"unknown chat template: {template}")


def strip_suffix_bos(tokens: list[int], bos: int | None) -> list[int]:
    if bos is None:
        return tokens
    if not tokens or tokens[0] != bos:
        raise RuntimeError("chat suffix BOS mismatch")
    return tokens[1:]


def prompt_body(stream_id: str, index: int, target_tokens: int) -> str:
    lead = (
        f"Mixed trace request {stream_id} {index}. Explain how an inference "
        "scheduler should place work across a desktop GPU, desktop CPU, and "
        "a phone. "
    )
    context = (
        "Discuss measured latency, energy, memory, transfer cost, execution "
        "overlap, model residency, and practical tradeoffs. Use concrete "
        "reasoning and complete sentences. "
    )
    repeats = max(128, (target_tokens + 15) // 16)
    return lead + context * repeats


def build_exact_prompt(
    codec: Codec,
    template: str,
    stream_id: str,
    index: int,
    target_tokens: int,
) -> tuple[list[int], str]:
    prefix, suffix, suffix_bos = chat_parts(template)
    body_tokens = codec.tokenize(
        prefix + prompt_body(stream_id, index, target_tokens)
    )
    suffix_tokens = strip_suffix_bos(codec.tokenize(suffix), suffix_bos)
    body_count = target_tokens - len(suffix_tokens)
    if body_count <= 0 or len(body_tokens) < body_count:
        raise RuntimeError("chat prompt target is too short or too long")
    tokens = body_tokens[:body_count] + suffix_tokens
    if len(tokens) != target_tokens:
        raise RuntimeError("chat prompt token conservation")
    return tokens, codec.detokenize(tokens)


def build_vlm_prompt(codec: Codec, question: str) -> tuple[list[int], str]:
    prefix, suffix, suffix_bos = chat_parts("gemma4")
    suffix_tokens = strip_suffix_bos(codec.tokenize(suffix), suffix_bos)
    tokens = codec.tokenize(prefix + question) + suffix_tokens
    return tokens, codec.detokenize(tokens)


def load_source() -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    source_bytes = SOURCE.read_bytes()
    if digest_bytes(source_bytes) != SOURCE_SHA256:
        raise RuntimeError("frozen source trace hash mismatch")
    rows: list[dict[str, Any]] = []
    raw_by_event: dict[str, bytes] = {}
    for index, raw in enumerate(source_bytes.splitlines(keepends=True)):
        if not raw.endswith(b"\n"):
            raise RuntimeError(f"source row {index} framing mismatch")
        row = json.loads(raw)
        if canonical(row) != raw:
            raise RuntimeError(f"source row {index} is not canonical")
        if row.get("schema") != SOURCE_SCHEMA or row.get("request_index") != index:
            raise RuntimeError(f"source row {index} identity mismatch")
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or event_id in raw_by_event:
            raise RuntimeError(f"source row {index} event identity")
        rows.append(row)
        raw_by_event[event_id] = raw
    if len(rows) != 74:
        raise RuntimeError("source record count mismatch")
    return rows, raw_by_event


def model_fields(model_id: str) -> dict[str, Any]:
    model = MODEL_INVENTORY[model_id]
    return {
        "model_artifact_bytes": model["artifact_bytes"],
        "model_artifact_sha256": model["artifact_sha256"],
    }


def original_rows(
    source_rows: list[dict[str, Any]],
    raw_by_event: dict[str, bytes],
) -> list[dict[str, Any]]:
    result = []
    for source_row in source_rows:
        execution_model_id = source_row["prompt_tokenizer_model"]
        if execution_model_id not in {
            "gemma-4-12b-it-q4_0",
            "qwen3-14b-q4_k_m",
        }:
            raise RuntimeError("source execution model identity")
        row = dict(source_row)
        row.update({
            "execution_model_id": execution_model_id,
            "modality": "text",
            "parent_record_sha256": digest_bytes(
                raw_by_event[source_row["event_id"]]
            ),
            "parent_schema": source_row["schema"],
            "prompt_transport": "tokens",
            "requested_model_id": execution_model_id,
            "schema": SCHEMA,
            "stream_request_index": source_row["request_index"],
            "text_input_tokens": source_row["input_tokens"],
            "trace_stream_id": "burstgpt-original",
        })
        row.update(model_fields(execution_model_id))
        result.append(row)
    return result


def text_overlay_rows(
    source_rows: list[dict[str, Any]],
    codecs: dict[str, Codec],
) -> list[dict[str, Any]]:
    result = []
    for stream in TEXT_STREAMS:
        codec = codecs[stream["tokenizer_model_id"]]
        for stream_index, donor_index in enumerate(DONOR_INDICES):
            donor = source_rows[donor_index]
            tokens, formatted_prompt_text = build_exact_prompt(
                codec,
                stream["template"],
                stream["stream_id"],
                stream_index,
                donor["input_tokens"],
            )
            model_id = stream["model_id"]
            row = {
                "arrival_us": donor["arrival_us"] + stream["arrival_offset_us"],
                "event_id": f"s42:{stream['stream_id']}:{stream_index:02d}",
                "execution_model_id": model_id,
                "formatted_prompt_text": formatted_prompt_text,
                "input_tokens": len(tokens),
                "modality": "text",
                "model_id": model_id,
                "output_tokens": donor["output_tokens"],
                "prompt_transport": "tokens",
                "prompt_tokenizer_model": stream["tokenizer_model_id"],
                "prompt_tokens": tokens,
                "requested_model_id": model_id,
                "schema": SCHEMA,
                "slo_us": donor["slo_us"],
                "source_input_tokens": donor["source_input_tokens"],
                "source_model": "BurstGPT-derived-overlay",
                "source_output_tokens": donor["source_output_tokens"],
                "source_shape_event_id": donor["event_id"],
                "source_shape_record_sha256": digest_bytes(canonical(donor)),
                "source_t_us": donor["source_t_us"],
                "stream_request_index": stream_index,
                "text_input_tokens": len(tokens),
                "trace_stream_id": stream["stream_id"],
            }
            row.update(model_fields(model_id))
            result.append(row)
    return result


def vlm_overlay_rows(
    source_rows: list[dict[str, Any]],
    codec: Codec,
) -> list[dict[str, Any]]:
    result = []
    model_id = "gemma-4-e2b-it-q8_0-vlm"
    model = MODEL_INVENTORY[model_id]
    for stream_index, (image_id, question, answer_groups) in enumerate(VLM_CASES):
        donor = source_rows[DONOR_INDICES[stream_index]]
        image = IMAGE_INVENTORY[image_id]
        tokens, formatted_prompt_text = build_vlm_prompt(codec, question)
        row = {
            "arrival_us": donor["arrival_us"] + 400_000,
            "event_id": f"s42:gemma-4-e2b-vlm-overlay:{stream_index:02d}",
            "execution_model_id": model_id,
            "formatted_prompt_text": formatted_prompt_text,
            "image": {
                "bytes": image["bytes"],
                "height": image["height"],
                "id": image_id,
                "media_type": image["media_type"],
                "path": image["path"],
                "sha256": image["sha256"],
                "width": image["width"],
            },
            "image_bytes": image["bytes"],
            "image_cache_key": (
                f"sha256:{image['sha256']}:"
                f"mmproj:{model['projector_sha256']}"
            ),
            "image_count": 1,
            "image_pixels": image["width"] * image["height"],
            "image_token_count_status": "runtime_measured",
            "input_tokens": len(tokens),
            "modality": "image_text",
            "model_id": model_id,
            "output_tokens": 32,
            "prompt_text": question,
            "prompt_transport": "multimodal_message",
            "prompt_tokenizer_model": model_id,
            "prompt_tokens": tokens,
            "quality_case": {
                "answer_groups": [list(group) for group in answer_groups],
                "case_id": f"s42-vlm-{stream_index:02d}",
                "question": question,
                "scorer": "normalized_contains_each_group_v1",
            },
            "requested_model_id": model_id,
            "schema": SCHEMA,
            "slo_us": donor["slo_us"],
            "source_input_tokens": donor["source_input_tokens"],
            "source_model": "BurstGPT-derived-vlm-overlay",
            "source_output_tokens": donor["source_output_tokens"],
            "source_shape_event_id": donor["event_id"],
            "source_shape_record_sha256": digest_bytes(canonical(donor)),
            "source_t_us": donor["source_t_us"],
            "stream_request_index": stream_index,
            "text_input_tokens": len(tokens),
            "trace_stream_id": "gemma-4-e2b-vlm-overlay",
            "vision_projector_bytes": model["projector_bytes"],
            "vision_projector_sha256": model["projector_sha256"],
        }
        row.update(model_fields(model_id))
        result.append(row)
    return result


def build_manifest(rows: list[dict[str, Any]], trace_bytes: bytes) -> dict[str, Any]:
    stream_counts = Counter(row["trace_stream_id"] for row in rows)
    model_counts = Counter(row["execution_model_id"] for row in rows)
    image_requests = [row for row in rows if row["modality"] == "image_text"]
    return {
        "arrival_first_us": rows[0]["arrival_us"],
        "arrival_last_us": rows[-1]["arrival_us"],
        "arrival_span_us": rows[-1]["arrival_us"] - rows[0]["arrival_us"],
        "image_assets": IMAGE_INVENTORY,
        "image_request_count": len(image_requests),
        "image_unique_count": len({row["image"]["sha256"] for row in image_requests}),
        "model_inventory": MODEL_INVENTORY,
        "models": dict(sorted(model_counts.items())),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "record_count": len(rows),
        "schema": MANIFEST_SCHEMA,
        "source": {
            "path": str(SOURCE.relative_to(REPO_ROOT)),
            "record_count": 74,
            "schema": SOURCE_SCHEMA,
            "sha256": SOURCE_SHA256,
        },
        "streams": dict(sorted(stream_counts.items())),
        "text_input_tokens": sum(row["text_input_tokens"] for row in rows),
        "trace": {
            "path": str(OUTPUT.relative_to(REPO_ROOT)),
            "schema": SCHEMA,
            "sha256": digest_bytes(trace_bytes),
        },
        "vlm_image_tokens": {
            "status": "runtime_measured",
            "value": None,
        },
    }


def write_or_compare(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"existing output differs: {path}")
        return
    path.write_bytes(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codec",
        type=Path,
        default=REPO_ROOT / "build-cuda/bin/llama-token-codec",
    )
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=REPO_ROOT / "build-cuda/bin",
    )
    parser.add_argument(
        "--qwen-model",
        type=Path,
        default=REPO_ROOT / "Qwen3-0.6B-Q8_0.gguf",
    )
    parser.add_argument(
        "--llama-model",
        type=Path,
        default=(
            REPO_ROOT.parent / "models/Llama-3.2-1B-Instruct-Q4_0.gguf"
        ),
    )
    parser.add_argument(
        "--vlm-model",
        type=Path,
        default=REPO_ROOT.parent / "models/gemma-4-E2B-it-Q8_0.gguf",
    )
    args = parser.parse_args()

    codec_models = {
        "qwen3-0.6b-q8_0": args.qwen_model,
        "llama-3.2-1b-instruct-q4_0": args.llama_model,
        "gemma-4-e2b-it-q8_0-vlm": args.vlm_model,
    }
    for model_id, path in codec_models.items():
        model = MODEL_INVENTORY[model_id]
        check_file(path, model["artifact_bytes"], model["artifact_sha256"])
    for image in IMAGE_INVENTORY.values():
        check_file(
            REPO_ROOT / image["path"],
            image["bytes"],
            image["sha256"],
        )

    source_rows, raw_by_event = load_source()
    codecs = {
        model_id: Codec(
            args.codec,
            path,
            MODEL_INVENTORY[model_id]["artifact_sha256"],
            args.library_dir,
        )
        for model_id, path in codec_models.items()
    }
    try:
        rows = original_rows(source_rows, raw_by_event)
        rows.extend(text_overlay_rows(source_rows, codecs))
        rows.extend(
            vlm_overlay_rows(
                source_rows,
                codecs["gemma-4-e2b-it-q8_0-vlm"],
            )
        )
    finally:
        for codec in codecs.values():
            codec.close()

    rows.sort(key=lambda row: (row["arrival_us"], row["event_id"]))
    if len(rows) != 114:
        raise RuntimeError("mixed trace record count")
    if len({row["event_id"] for row in rows}) != len(rows):
        raise RuntimeError("mixed trace event collision")
    for mixed_index, row in enumerate(rows):
        row["mixed_request_index"] = mixed_index
        if len(row["prompt_tokens"]) != row["input_tokens"]:
            raise RuntimeError("mixed trace prompt token conservation")
    trace_bytes = b"".join(canonical(row) for row in rows)
    manifest = build_manifest(rows, trace_bytes)
    manifest_bytes = canonical(manifest)
    write_or_compare(OUTPUT, trace_bytes)
    write_or_compare(MANIFEST, manifest_bytes)
    print(json.dumps({
        "manifest": str(MANIFEST),
        "models": manifest["models"],
        "output_tokens": manifest["output_tokens"],
        "records": manifest["record_count"],
        "text_input_tokens": manifest["text_input_tokens"],
        "trace": str(OUTPUT),
        "trace_sha256": manifest["trace"]["sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
