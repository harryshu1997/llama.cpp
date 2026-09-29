#!/usr/bin/env python3
"""Build a bounded BurstGPT trace with target-model chat prompt tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


SOURCE_SHA256 = (
    "94c36fe3ac43281dc0c83a29a1519c7"
    "e72ac445ed9ded98d474b2231041c1735"
)
HOT_MODEL_SHA256 = (
    "500a8806e85ee9c83f3ae084202955924"
    "51379b4f8cf2d0f41c15dffeb6b81f0"
)
COLD_MODEL_SHA256 = (
    "494518c2262a26e2a607af0e40bca11c"
    "4de5a0b108e7c21308906dbbfb1c6f8c"
)
SOURCE_HOT = "gemma-4-12b-it-q8_0"
SOURCE_COLD = "qwen3-14b-q4_k_m"
PREFILL_CAP = 512
DECODE_CAP = 32


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


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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
                "--model", str(model),
                "--model-sha256", model_sha256,
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
        return tokens

    def validate_detokenize(self, tokens: list[int]) -> None:
        response = self.exchange("detokenize", "tokens", tokens)
        if not isinstance(response.get("text"), str):
            raise RuntimeError("token codec detokenize response")

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


def prompt_body(index: int, target_tokens: int) -> str:
    lead = (
        f"Request {index}. Explain in clear English how a desktop computer "
        "and a phone can cooperate on neural-network inference. "
    )
    context = (
        "Discuss measured latency, memory bandwidth, computation, overlap, "
        "and practical tradeoffs. Use concrete reasoning and complete "
        "sentences. "
    )
    repeats = max(128, (target_tokens + 15) // 16)
    return lead + context * repeats


def build_prompt(
    codec: Codec,
    index: int,
    target_tokens: int,
    prefix: str,
    suffix: str,
    suffix_bos: int | None,
) -> list[int]:
    body_tokens = codec.tokenize(prefix + prompt_body(index, target_tokens))
    suffix_tokens = codec.tokenize(suffix)
    if suffix_bos is not None:
        if not suffix_tokens or suffix_tokens[0] != suffix_bos:
            raise RuntimeError("chat suffix BOS mismatch")
        suffix_tokens = suffix_tokens[1:]
    body_count = target_tokens - len(suffix_tokens)
    if body_count <= 0 or len(body_tokens) < body_count:
        raise RuntimeError("chat prompt target is too short or too long")
    result = body_tokens[:body_count] + suffix_tokens
    if len(result) != target_tokens:
        raise RuntimeError("chat prompt token conservation")
    codec.validate_detokenize(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--hot-model", type=Path, required=True)
    parser.add_argument("--cold-model", type=Path, required=True)
    parser.add_argument("--prefill-cap", type=int, default=PREFILL_CAP)
    parser.add_argument("--decode-cap", type=int, default=DECODE_CAP)
    parser.add_argument(
        "--schema",
        default="s41-gemma-qwen-request-semantic-long-v1",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("output already exists")
    if args.prefill_cap <= 0 or args.decode_cap <= 0:
        raise RuntimeError("token caps must be positive")
    try:
        args.schema.encode("ascii")
    except UnicodeEncodeError as error:
        raise RuntimeError("schema must be ASCII") from error
    if not args.schema:
        raise RuntimeError("schema must be non-empty")
    if digest_file(args.source) != SOURCE_SHA256:
        raise RuntimeError("frozen source trace hash mismatch")
    if (
        args.hot_model.stat().st_size != 9_001_752_960
        or digest_file(args.hot_model) != HOT_MODEL_SHA256
    ):
        raise RuntimeError("hot model identity mismatch")
    if (
        args.cold_model.stat().st_size != 6_975_878_176
        or digest_file(args.cold_model) != COLD_MODEL_SHA256
    ):
        raise RuntimeError("cold model identity mismatch")

    hot_codec = Codec(
        args.codec,
        args.hot_model,
        HOT_MODEL_SHA256,
        args.library_dir,
    )
    cold_codec = Codec(
        args.codec,
        args.cold_model,
        COLD_MODEL_SHA256,
        args.library_dir,
    )
    output = bytearray()
    try:
        for index, raw in enumerate(
            args.source.read_bytes().splitlines(keepends=True)
        ):
            if not raw.endswith(b"\n"):
                raise RuntimeError(f"source row {index} framing mismatch")
            row = json.loads(raw)
            if canonical(row) != raw:
                raise RuntimeError(f"source row {index} is not canonical")
            input_tokens = min(
                row["source_input_tokens"], args.prefill_cap
            )
            output_tokens = min(
                row["source_output_tokens"], args.decode_cap
            )
            if row["model_id"] == SOURCE_HOT:
                codec = hot_codec
                prompt = build_prompt(
                    codec,
                    index,
                    input_tokens,
                    "<|im_start|>user\n",
                    "\n<|im_end|>\n<|im_start|>assistant\n",
                    None,
                )
                tokenizer_model = "qwen3-14b-q4_k_m"
            elif row["model_id"] == SOURCE_COLD:
                codec = cold_codec
                prompt = build_prompt(
                    codec,
                    index,
                    input_tokens,
                    "<|turn>user\n",
                    "\n<turn|>\n<|turn>model\n"
                    "<|channel>thought\n<channel|>",
                    2,
                )
                tokenizer_model = "gemma-4-12b-it-q4_0"
            else:
                raise RuntimeError(f"source row {index} model")
            row["input_tokens"] = input_tokens
            row["output_tokens"] = output_tokens
            row["prompt_tokens"] = prompt
            row["prompt_tokenizer_model"] = tokenizer_model
            row["schema"] = args.schema
            output.extend(canonical(row))
    finally:
        hot_codec.close()
        cold_codec.close()

    args.output.write_bytes(output)
    print(json.dumps({
        "decode_cap": args.decode_cap,
        "output": str(args.output),
        "prefill_cap": args.prefill_cap,
        "records": len(output.splitlines()),
        "sha256": hashlib.sha256(output).hexdigest(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
