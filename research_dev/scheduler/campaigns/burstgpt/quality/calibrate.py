#!/usr/bin/env python3
"""Off-rig calibration of the inputs to the quality margin (GSM8K train split, one GPU per llama-server).

The rig compares the all-desktop arm with the phone-assisted arms on identical weights (Q4 checkpoints
dequantized to f16), so it measures only the offloading penalty. This module measures, separately and off the
rig, the two differences that the rig comparison cannot see, plus the noise floor they must be read against:

    quantization       original (f16 / bf16) vs the matched Q4 checkpoint        (quantization penalty)
    execution-format   Q4 vs the same Q4 weights dequantized to f16 (rig file)   (execution-format difference)
    noise-floor        the dequantized f16 run vs a repeat of itself              (batched greedy run-to-run noise)

Items come from the GSM8K TRAIN split only (the test split is the confirmatory study's), minus every item of the
WS4 pilot (`build_trace.assign_items` with the pilot seed), in one seeded permutation split between the two
models (disjoint items, as in the confirmatory design). Prompts, the 512-token budget with end of generation
banned, greedy decoding and the earliest-final-answer rule are the protocol's.

One deliberate difference from the rig's request body: `stream` is false. llama-server streams no chunk for a
token whose text ends inside a multi-byte UTF-8 character, and the chunk that completes the character carries
only its own token id, so a streamed output that contains such a character (Qwen's " \u2705" is two tokens)
has fewer token ids than n_predict. The non-streamed response returns every token id and the same text;
sampling is unaffected. Per-token character offsets (for answer positions and shorter-budget readouts) are
rebuilt offline from the GGUF vocabulary and used only where the rebuilt text equals the server's text.

    python3 -m research_dev.scheduler.campaigns.burstgpt.quality.calibrate run \\
        --gsm8k gsm8k_train.jsonl --role qwen --format q4 --model Qwen3-14B-Q4_K_M.gguf \\
        --server build-cuda-s43/bin/llama-server --gpu 0 --port 18120 --items 200 \\
        --prompts OUT/prompts-qwen.jsonl --write-prompts --run-dir OUT/runs/qwen-q4

    python3 -m ...quality.calibrate size --gsm8k ... --run qwen:original=DIR ... --out SIZE.json
    python3 -m ...quality.calibrate score --gsm8k ... --run qwen:original=DIR ... \\
        --tokenizer-model qwen=Qwen3-14B-Q4_K_M.gguf --tokenizer-model gemma=gemma-4-12B-it-Q4_0.gguf \\
        --qwen-items N --gemma-items N --out REPORT.json --md REPORT.md

`run` is resumable and extendable: rerunning with a larger --items only sends the missing items. `size` is the
blinded interim: it prints discordance totals only (no accuracies, no signs) and the sample size rule's N per
model. `score` writes the full report (per-format accuracy, paired differences with Tango and Newcombe
intervals, discordance, extraction and truncation, output similarity), per model and pooled.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import http.client
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import protocol
from .build_trace import ROLES as PILOT_ROLES, assign_items
from .gsm8k import (
    EXPECTED_FIRST_TOKENS, EXPECTED_LAST_TOKENS, Gsm8kItem, extract_answer, load_items,
    render_prompt,
)
from .score import KeyRow, Output, similarity, truncated_extraction
from .stats import (
    PairedCounts, exact_mcnemar_p, newcombe_interval, noninferiority_power, noninferiority_sample_size_normal,
    tango_interval, z_quantile,
)

CALIBRATION_ID = "ws9-gsm8k-calibration-v1"
CALIBRATION_SEED = 20261001
ROLES = ("qwen", "gemma")
FORMATS = ("original", "q4", "dequant", "dequant-repeat", "q4-local")
# (name, baseline format, treatment format); d = treatment accuracy - baseline accuracy
COMPARISONS = (
    ("quantization", "original", "q4"),
    ("execution-format", "q4", "dequant"),
    ("noise-floor", "dequant", "dequant-repeat"),
)
# descriptive only: the rig's weights vs the original; and, where a `q4-local` run exists (a Q4 quantized here from
# the original file, for a model whose shipped Q4 came from a different source checkpoint), the naive quantization
# penalty and the effect of the shipped Q4's source
DESCRIPTIVE_COMPARISONS = (
    ("rig-weights", "original", "dequant"),
    ("quantization-local", "original", "q4-local"),
    ("q4-source", "q4-local", "q4"),
)
SIZING_COMPARISONS = tuple(name for name, _, _ in COMPARISONS)
OUTPUT_TOKENS = protocol.PILOT_OUTPUT_TOKENS
PILOT_ITEMS = 200
TARGET_HALF_WIDTH = 0.015
CONFIDENCE = 0.95
MAXIMUM_ITEMS = 3000
ITEMS_ROUNDING = 100
PARALLEL = 32
SLOT_CONTEXT = 1024
# identical llama-server flags for every format (the model path, GPU and port are the only per-run inputs)
SERVER_FLAGS = ("-ngl", "999", "-fit", "off", "-np", str(PARALLEL), "-c", str(PARALLEL * SLOT_CONTEXT),
                "-cram", "0")
RUN_SCHEMA = "research-scheduler-quality-calibration-run-v1"
PROMPTS_SCHEMA = "research-scheduler-quality-calibration-prompts-v1"
SIZE_SCHEMA = "research-scheduler-quality-calibration-size-v1"
REPORT_SCHEMA = "research-scheduler-quality-calibration-report-v1"
# planning readout for the confirmatory study (descriptive; the margin itself is not chosen here)
PLANNING_MARGINS = (0.01, 0.015, 0.02, 0.03)
PLANNING_PAIRS = (512, 768)
REQUEST_ATTEMPTS = 3
REQUEST_TIMEOUT_S = 3600.0
REPO_ROOT = Path(__file__).resolve().parents[5]


class CalibrationError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------- items


def pilot_item_indices(items: list[Gsm8kItem]) -> set[int]:
    """Split indices of every item of the WS4 pilot (`build_trace --pilot`: one shard of PILOT_PER_MODEL items per
    role from the PILOT_SEED permutation of the train split), reproduced with the builder's own assignment."""
    shard = assign_items(items, seed=protocol.PILOT_SEED, shards=1,
                         per_shard={role: protocol.PILOT_PER_MODEL for role in PILOT_ROLES})[0]
    return {item.index for role in PILOT_ROLES for item in shard[role]}


def calibration_sequences(items: list[Gsm8kItem], seed: int = CALIBRATION_SEED) -> dict[str, list[Gsm8kItem]]:
    """Per-model item sequences: the split minus the pilot, one seeded permutation, Qwen at the even positions and
    Gemma at the odd ones (disjoint). A run of N items uses the first N of its model's sequence."""
    if any(item.split != "train" for item in items):
        raise CalibrationError("calibration uses the GSM8K train split only")
    excluded = pilot_item_indices(items)
    order = [item.index for item in items if item.index not in excluded]
    random.Random(seed).shuffle(order)
    by_index = {item.index: item for item in items}
    return {"qwen": [by_index[index] for index in order[0::2]],
            "gemma": [by_index[index] for index in order[1::2]]}


def required_items(discordance: float, *, half_width: float = TARGET_HALF_WIDTH, confidence: float = CONFIDENCE,
                   minimum: int = PILOT_ITEMS, maximum: int = MAXIMUM_ITEMS, rounding: int = ITEMS_ROUNDING) -> int:
    """Items for a paired-difference CI half-width <= half_width at d = 0: n = z^2 psi / h^2 (the variance of the
    paired mean difference is (psi - d^2) / n <= psi / n), rounded up to `rounding`, within [minimum, maximum]."""
    if not 0.0 <= discordance <= 1.0 or half_width <= 0.0 or not 0.0 < confidence < 1.0:
        raise CalibrationError("sizing parameters are invalid")
    z = z_quantile(1.0 - (1.0 - confidence) / 2.0)
    raw = z * z * discordance / (half_width * half_width)
    rounded = rounding * math.ceil(raw / rounding) if raw > 0 else 0
    return min(maximum, max(minimum, rounded))


def expected_half_width(discordance: float, n: int, confidence: float = CONFIDENCE) -> float:
    return z_quantile(1.0 - (1.0 - confidence) / 2.0) * math.sqrt(discordance / n)


# ----------------------------------------------------------------------------------------------- hashing


def sha256_file(path: Path, chunk: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def tokens_sha256(tokens: list[int]) -> str:
    """The runner's prompt_sha256 of the same tokens (build_trace.prompt_sha256)."""
    return "sha256:" + hashlib.sha256(canonical(tokens)).hexdigest()


def item_list_sha256(items: list[Gsm8kItem]) -> str:
    return "sha256:" + hashlib.sha256(canonical([item.item_id for item in items])).hexdigest()


# ----------------------------------------------------------------------------------------------- requests


def request_body(tokens: list[int], seed: int, output_tokens: int = OUTPUT_TOKENS) -> dict[str, Any]:
    """The rig's /completion body (adapters/http_backend.py LlamaCppHttpClient.complete) with stream false (see
    the module docstring: a streamed body loses the token ids of characters split across tokens)."""
    return {"cache_prompt": False, "ignore_eos": True, "n_predict": output_tokens, "prompt": list(tokens),
            "return_tokens": True, "seed": seed, "stream": False, "temperature": 0.0}


def response_output(raw: bytes, output_tokens: int = OUTPUT_TOKENS) -> tuple[list[int], str]:
    """Token ids and text of one non-streamed /completion response; every generated token must be present."""
    value = json.loads(raw)
    tokens, content = value.get("tokens"), value.get("content")
    if type(tokens) is not list or not all(type(token) is int for token in tokens) or type(content) is not str:
        raise CalibrationError("completion response is malformed")
    if len(tokens) != output_tokens or value.get("tokens_predicted") != output_tokens:
        raise CalibrationError(f"completion has {len(tokens)} token ids / {value.get('tokens_predicted')} predicted, "
                               f"not {output_tokens}")
    return tokens, content


def check_prompt_structure(role: str, tokens: list[int]) -> None:
    first, last = EXPECTED_FIRST_TOKENS[role], EXPECTED_LAST_TOKENS[role]
    if tuple(tokens[:len(first)]) != first or tuple(tokens[-len(last):]) != last:
        raise CalibrationError(f"{role} prompt tokens do not carry the chat-template structure: starts "
                               f"{tokens[:len(first)]}, ends {tokens[-len(last):]}")


def _gpt2_byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode (llama.cpp unicode_utf8_to_byte)."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1)) + list(
        range(ord("\u00ae"), ord("\u00ff") + 1))
    codes = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + extra)
            extra += 1
    return {chr(code): byte for byte, code in zip(printable, codes)}


class PieceTable:
    """Bytes of each token as llama-server renders it into `content` (llama_vocab::token_to_piece with
    special = false): control / unknown / unused tokens render empty, user-defined tokens as their text, byte
    tokens as their byte, normal tokens GPT-2 byte-decoded (BPE, e.g. Qwen) or with U+2581 -> space (Gemma)."""

    CONTROL, USER_DEFINED, UNUSED, BYTE, UNKNOWN = 3, 4, 5, 6, 2

    def __init__(self, texts: list[str], types: list[int], model: str):
        self.texts, self.types, self.model = texts, types, model
        self.byte_level = model == "gpt2"
        self._decoder = _gpt2_byte_decoder() if self.byte_level else None
        self._cache: dict[int, bytes] = {}

    @classmethod
    def from_gguf(cls, path: Path) -> "PieceTable":
        if str(REPO_ROOT / "gguf-py") not in sys.path:
            sys.path.insert(0, str(REPO_ROOT / "gguf-py"))
        from gguf import GGUFReader  # noqa: E402 (repository package, imported on demand)
        reader = GGUFReader(str(path))
        field = reader.fields["tokenizer.ggml.tokens"]
        texts = [bytes(field.parts[index]).decode("utf-8") for index in field.data]
        field = reader.fields["tokenizer.ggml.token_type"]
        types = [int(field.parts[index][0]) for index in field.data]
        field = reader.fields["tokenizer.ggml.model"]
        model = bytes(field.parts[field.data[0]]).decode("utf-8")
        return cls(texts, types, model)

    def piece(self, token: int) -> bytes:
        if token in self._cache:
            return self._cache[token]
        kind, text = self.types[token], self.texts[token]
        if kind in (self.CONTROL, self.UNKNOWN, self.UNUSED):
            value = b""
        elif kind == self.USER_DEFINED:
            value = text.encode("utf-8")
        elif kind == self.BYTE:
            value = bytes([int(text[3:-1], 16)])
        elif self.byte_level:
            value = bytes(self._decoder[char] if char in self._decoder else 0 for char in text) if all(
                char in self._decoder for char in text) else text.encode("utf-8")
        else:
            value = text.replace("\u2581", " ").encode("utf-8")
        self._cache[token] = value
        return value

    def char_starts(self, tokens: list[int], content: str) -> list[tuple[int, int]] | None:
        """(token index, character offset of the complete characters before it) for every token, or None when the
        rebuilt text differs from the server's content (then no position readout uses this output)."""
        data = bytearray()
        byte_starts = []
        for token in tokens:
            byte_starts.append(len(data))
            data += self.piece(token)
        if bytes(data).decode("utf-8", "replace") != content:
            return None
        return [(index, len(bytes(data[:start]).decode("utf-8", "ignore"))) for index, start in enumerate(byte_starts)]


class LlamaServer:
    """One llama-server on one GPU with the fixed SERVER_FLAGS; terminated on exit."""

    def __init__(self, binary: Path, model: Path, gpu: str, port: int, log_path: Path):
        self.command = [str(binary), "-m", str(model), *SERVER_FLAGS, "--host", "127.0.0.1", "--port", str(port)]
        self.port, self.gpu, self.log_path = port, gpu, log_path
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> "LlamaServer":
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpu)
        self.log = self.log_path.open("ab")
        self.process = subprocess.Popen(self.command, stdout=self.log, stderr=subprocess.STDOUT, env=environment,
                                        cwd=str(self.log_path.parent))
        try:
            self._wait_healthy()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _wait_healthy(self, timeout_s: float = 900.0) -> None:
        """/health answers 503 while the model loads and 200 {"status": "ok"} once it serves."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise CalibrationError(f"llama-server exited with {self.process.returncode}; see {self.log_path}")
            try:
                if self.get("/health").get("status") == "ok":
                    return
            except (OSError, http.client.HTTPException, ValueError, CalibrationError):
                pass
            time.sleep(2.0)
        raise CalibrationError(f"llama-server did not become healthy within {timeout_s:.0f} s")

    def __exit__(self, *exc) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()

    def _connection(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=REQUEST_TIMEOUT_S)

    def get(self, path: str) -> dict[str, Any]:
        connection = self._connection()
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            body = response.read()
            if response.status != 200:
                raise CalibrationError(f"GET {path} status {response.status}")
            return json.loads(body)
        finally:
            connection.close()

    def post(self, path: str, body: dict[str, Any]) -> bytes:
        connection = self._connection()
        try:
            connection.request("POST", path, body=json.dumps(body, sort_keys=True),
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            data = response.read()
            if response.status != 200:
                raise CalibrationError(f"POST {path} status {response.status}: {data[:200]!r}")
            return data
        finally:
            connection.close()

    def tokenize(self, text: str) -> list[int]:
        value = json.loads(self.post("/tokenize", {"content": text, "add_special": True, "parse_special": True}))
        tokens = value.get("tokens")
        if type(tokens) is not list or not all(type(token) is int for token in tokens):
            raise CalibrationError("tokenize response is malformed")
        return tokens


def gpu_inventory() -> dict[str, Any]:
    """nvidia-smi view of the GPUs and every compute process (to refuse a GPU someone else uses)."""
    def query(arguments: list[str]) -> list[list[str]]:
        text = subprocess.run(["nvidia-smi", *arguments, "--format=csv,noheader,nounits"], check=True,
                              capture_output=True, text=True).stdout
        return [[field.strip() for field in line.split(",")] for line in text.splitlines() if line.strip()]
    gpus = [{"index": row[0], "uuid": row[1], "name": row[2], "memory_used_mib": int(row[3]),
             "driver": row[4]} for row in query(["--query-gpu=index,uuid,name,memory.used,driver_version"])]
    apps = [{"pid": int(row[0]), "gpu_uuid": row[1], "used_memory_mib": row[2]}
            for row in query(["--query-compute-apps=pid,gpu_uuid,used_memory"])]
    return {"gpus": gpus, "compute_apps": apps}


def load_prompts(path: Path) -> dict[str, list[int]]:
    prompts = {}
    for line in path.read_text(encoding="ascii").splitlines():
        row = json.loads(line)
        prompts[row["item_id"]] = row["tokens"]
    return prompts


def run(args: argparse.Namespace) -> dict[str, Any]:
    items = load_items(args.gsm8k, "train")
    sequence = calibration_sequences(items)[args.role]
    if not 1 <= args.items <= len(sequence):
        raise CalibrationError(f"--items must be 1..{len(sequence)}")
    selected = sequence[:args.items]
    run_dir = args.run_dir
    streams = run_dir / "streams"
    streams.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "RUN.json"
    previous = json.loads(manifest_path.read_text(encoding="ascii")) if manifest_path.is_file() else None
    inventory = gpu_inventory()
    target = next((gpu for gpu in inventory["gpus"] if gpu["index"] == args.gpu), None)
    if target is None:
        raise CalibrationError(f"GPU {args.gpu} not found")
    others = [app for app in inventory["compute_apps"] if app["gpu_uuid"] == target["uuid"]]
    if others and not args.allow_busy_gpu:
        raise CalibrationError(f"GPU {args.gpu} has compute processes {others}; choose a free GPU")
    model_sha256 = args.model_sha256 or sha256_file(args.model)
    if args.model_sha256 and args.verify_model_sha256 and sha256_file(args.model) != args.model_sha256:
        raise CalibrationError("model sha256 differs from --model-sha256")
    binary_dir = args.server.resolve().parent
    binaries = {path.name: sha256_file(path) for path in sorted(binary_dir.iterdir())
                if path.is_file() and (path.name == args.server.name or path.suffix == ".so" or ".so." in path.name)}
    identity = {"calibration_id": CALIBRATION_ID, "role": args.role, "format": args.format,
                "model_file": str(args.model), "model_sha256": model_sha256, "server_flags": list(SERVER_FLAGS),
                "server_binary": str(args.server), "server_binaries_sha256": binaries,
                "output_tokens": OUTPUT_TOKENS, "parallel_requests": PARALLEL, "calibration_seed": CALIBRATION_SEED}
    if previous is not None:
        changed = [name for name, value in identity.items() if previous.get("identity", {}).get(name) != value]
        if changed:
            raise CalibrationError(f"{run_dir} was run with different settings: {changed}")
    reference_prompts = load_prompts(args.prompts) if args.prompts.is_file() else None
    if reference_prompts is None and not args.write_prompts:
        raise CalibrationError(f"{args.prompts} does not exist (pass --write-prompts on the tokenizer-model run)")
    session: dict[str, Any] = {"started_utc": _now(), "items_requested": args.items, "gpu": target,
                               "other_compute_apps": others}
    failures: list[dict[str, Any]] = []
    lock = threading.Lock()
    with LlamaServer(args.server, args.model, args.gpu, args.port, run_dir / "server.log") as server:
        props = server.get("/props")
        session["server_props"] = {"build_info": props.get("build_info"), "total_slots": props.get("total_slots"),
                                   "n_ctx_slot": (props.get("default_generation_settings") or {}).get("n_ctx")}
        tokenized = {}
        for item in selected:
            tokens = server.tokenize(render_prompt(args.role, item.question))
            check_prompt_structure(args.role, tokens)
            tokenized[item.item_id] = tokens
        if reference_prompts is None or any(item.item_id not in reference_prompts for item in selected):
            if not args.write_prompts:
                raise CalibrationError("the reference prompts do not cover these items (extend them with the "
                                       "tokenizer-model run and --write-prompts first)")
            merged = dict(reference_prompts or {})
            for item in selected:
                merged.setdefault(item.item_id, tokenized[item.item_id])
            order = [entry.item_id for entry in sequence if entry.item_id in merged]
            args.prompts.write_bytes(b"".join(canonical({"schema": PROMPTS_SCHEMA, "item_id": item_id,
                                                         "role": args.role, "tokens": merged[item_id]})
                                              for item_id in order))
            reference_prompts = merged
        mismatched = [item.item_id for item in selected if tokenized[item.item_id] != reference_prompts[item.item_id]]
        session["own_tokenization_mismatches"] = mismatched
        if mismatched and not args.allow_tokenizer_mismatch:
            raise CalibrationError(f"{len(mismatched)} prompts tokenize differently from the reference tokenizer")
        pending = [item for item in selected if not (streams / f"{item.item_id}.json").is_file()]
        session["items_sent"] = len(pending)

        def one(item: Gsm8kItem) -> None:
            body = request_body(reference_prompts[item.item_id], item.index)
            last_error = None
            for attempt in range(1, REQUEST_ATTEMPTS + 1):
                try:
                    raw = server.post("/completion", body)
                    response_output(raw)
                    partial = streams / f"{item.item_id}.json.part"
                    partial.write_bytes(raw)
                    partial.replace(streams / f"{item.item_id}.json")
                    return
                except (OSError, http.client.HTTPException, CalibrationError, ValueError) as error:
                    last_error = f"attempt {attempt}: {type(error).__name__}: {error}"
            with lock:
                failures.append({"item_id": item.item_id, "error": last_error})

        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            list(pool.map(one, pending))
        session["request_seconds"] = round(time.monotonic() - started, 1)
    session["ended_utc"] = _now()
    session["failures"] = failures
    complete = {item.item_id for item in sequence if (streams / f"{item.item_id}.json").is_file()}
    prefix = 0
    while prefix < len(sequence) and sequence[prefix].item_id in complete:
        prefix += 1
    manifest = {"schema": RUN_SCHEMA, "identity": identity,
                "sessions": [*(previous or {}).get("sessions", []), session],
                "complete_prefix_items": prefix, "complete_items": len(complete),
                "prompts_file": str(args.prompts), "prompts_sha256": sha256_file(args.prompts),
                "item_list_sha256_of_prefix": item_list_sha256(sequence[:prefix])}
    manifest_path.write_bytes(canonical(manifest))
    if failures:
        raise CalibrationError(f"{len(failures)} requests failed; rerun to resume ({run_dir})")
    return {"run_dir": str(run_dir), "complete_prefix_items": prefix, "request_seconds": session["request_seconds"]}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------------------------- scoring


@dataclass(frozen=True)
class RunSpec:
    role: str
    format: str
    directory: Path


def parse_run(value: str) -> RunSpec:
    """ROLE:FORMAT=DIR"""
    label, _, directory = value.partition("=")
    role, _, fmt = label.partition(":")
    if role not in ROLES or fmt not in FORMATS or not directory:
        raise argparse.ArgumentTypeError(f"--run expects ROLE:FORMAT=DIR with ROLE in {ROLES}, FORMAT in {FORMATS}")
    return RunSpec(role, fmt, Path(directory))


@dataclass
class CalibrationOutput(Output):
    """A scored output; `positions` is True when per-token character offsets were rebuilt exactly."""

    positions: bool = False
    tolerant: bool = False


# descriptive sensitivity only (never the frozen rule): an extracted answer within this relative distance of the
# reference counts as right, e.g. "Final answer: 46.00000000000001" for 46 (a floating-point formatting artifact)
TOLERANCE = 1e-6


def tolerant_match(value, reference) -> bool:
    return value is not None and abs(value - reference) <= Decimal(str(TOLERANCE)) * max(Decimal(1), abs(reference))


def load_outputs(spec: RunSpec, items: list[Gsm8kItem], pieces: PieceTable | None = None
                 ) -> dict[str, CalibrationOutput]:
    """Scored outputs of the first len(items) items of a run; an absent or incomplete output is an error."""
    outputs = {}
    for item in items:
        path = spec.directory / "streams" / f"{item.item_id}.json"
        if not path.is_file():
            raise CalibrationError(f"{spec.role}:{spec.format} has no output for {item.item_id} ({spec.directory})")
        tokens, text = response_output(path.read_bytes())
        starts = pieces.char_starts(tokens, text) if pieces is not None else None
        key = KeyRow(request_id=item.item_id, trace_name=CALIBRATION_ID, shard=0, role=spec.role,
                     item_id=item.item_id, reference=item.reference, combined_request_index=item.index,
                     prompt_sha256="", output_tokens=OUTPUT_TOKENS)
        output = CalibrationOutput(key=key, tokens=tokens, text=text, chunk_starts=starts or [(0, 0)],
                                   positions=starts is not None)
        output.extraction = extract_answer(text)
        output.correct = output.extraction.found and output.extraction.value == item.reference
        output.tolerant = tolerant_match(output.extraction.value, item.reference)
        outputs[item.item_id] = output
    return outputs


def _quantiles(values: list[int]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]  # noqa: E731
    return {"min": ordered[0], "p50": pick(0.5), "p90": pick(0.9), "p99": pick(0.99), "max": ordered[-1],
            "mean": sum(ordered) / len(ordered)}


def format_summary(outputs: list[CalibrationOutput]) -> dict[str, Any]:
    """Accuracy, extraction and truncation of one run. The 512-token readouts use the full text; the shorter-budget
    misses and the answer positions use only outputs whose token offsets were rebuilt exactly."""
    n = len(outputs)
    rules: dict[str, int] = {}
    for output in outputs:
        rules[output.extraction.rule] = rules.get(output.extraction.rule, 0) + 1
    correct = sum(output.correct for output in outputs)
    found = [output for output in outputs if output.extraction.found]
    located = [output for output in outputs if output.positions]
    misses = {str(budget): sum(not truncated_extraction(output, budget).found for output in located)
              for budget in protocol.OUTPUT_TOKEN_CANDIDATES}
    low, high = _wilson(correct, n)
    return {"n": n, "correct": correct, "accuracy": correct / n, "accuracy_wilson_95": [low, high],
            "extraction_rules": dict(sorted(rules.items())), "extracted": len(found),
            "extraction_rate": len(found) / n, "final_answer_rate": rules.get("final-answer", 0) / n,
            "no_answer_within_budget": n - len(found), "positions_available": len(located),
            "float_near_misses": sum(output.tolerant and not output.correct for output in outputs),
            "tolerant_correct": sum(output.tolerant for output in outputs),
            "misses_by_budget_of_positioned": misses,
            "answer_token_end": _quantiles([output.answer_token_end() for output in found if output.positions])}


def _wilson(successes: int, n: int) -> tuple[float, float]:
    z = z_quantile(1.0 - (1.0 - CONFIDENCE) / 2.0)
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def paired(baseline: dict[str, Output], treatment: dict[str, Output], item_ids: list[str],
           field: str = "correct") -> PairedCounts:
    return PairedCounts.from_pairs((getattr(baseline[item_id], field), getattr(treatment[item_id], field))
                                   for item_id in item_ids)


def comparison_summary(counts: PairedCounts) -> dict[str, Any]:
    tango = tango_interval(counts, CONFIDENCE)
    newcombe = newcombe_interval(counts, CONFIDENCE)
    return {**counts.to_json(), "discordance": counts.discordant / counts.n,
            "tango_95": [tango[0], tango[1]], "newcombe_95": [newcombe[0], newcombe[1]],
            "tango_half_width": (tango[1] - tango[0]) / 2.0,
            "exact_mcnemar_p_two_sided": exact_mcnemar_p(counts)}


def planning(discordance: float) -> dict[str, Any]:
    """What a measured discordance implies for the confirmatory Tango test (true d = 0, one-sided alpha 0.025):
    asymptotic pairs for power 0.8 per candidate margin, and the exact power at the planned / capped pair counts."""
    half = discordance / 2.0
    return {"discordance": discordance,
            "pairs_for_power_asymptotic": {
                f"{margin:.3f}": noninferiority_sample_size_normal(half, half, margin, protocol.ALPHA_ONE_SIDED,
                                                                   protocol.TARGET_POWER)
                for margin in PLANNING_MARGINS},
            "exact_power": {str(pairs): {f"{margin:.3f}": noninferiority_power(pairs, half, half, margin,
                                                                                protocol.ALPHA_ONE_SIDED)
                                         for margin in PLANNING_MARGINS}
                            for pairs in PLANNING_PAIRS},
            "target_power": protocol.TARGET_POWER, "alpha_one_sided": protocol.ALPHA_ONE_SIDED}


def similarity_counts(baseline: dict[str, Output], treatment: dict[str, Output], item_ids: list[str]) -> dict:
    values = [similarity(baseline[item_id], treatment[item_id]) for item_id in item_ids]
    diverged = [value["first_divergence"] for value in values if not value["identical"]]
    n = len(values)
    return {"pairs": n, "identical_share": sum(value["identical"] for value in values) / n,
            "identical_through_answer_share": sum(value["identical_through_answer"] for value in values) / n,
            "same_answer_share": sum(value["same_answer"] for value in values) / n,
            "first_divergence_of_diverged": _quantiles(diverged)}


def _load_all(specs: list[RunSpec], sequences: dict[str, list[Gsm8kItem]], limit: dict[str, int],
              pieces: dict[str, PieceTable] | None = None) -> dict[tuple[str, str], dict[str, CalibrationOutput]]:
    runs = {}
    for spec in specs:
        if (spec.role, spec.format) in runs:
            raise CalibrationError(f"two runs for {spec.role}:{spec.format}")
        runs[(spec.role, spec.format)] = load_outputs(spec, sequences[spec.role][:limit[spec.role]],
                                                      (pieces or {}).get(spec.role))
    return runs


def _check_identities(specs: list[RunSpec]) -> dict[str, Any]:
    """Every run of a model used the same prompts file content, flags, budget and binaries."""
    identities = {}
    for spec in specs:
        manifest = json.loads((spec.directory / "RUN.json").read_text(encoding="ascii"))
        identity = manifest["identity"]
        if identity["role"] != spec.role or identity["format"] != spec.format:
            raise CalibrationError(f"{spec.directory} is {identity['role']}:{identity['format']}, not "
                                   f"{spec.role}:{spec.format}")
        identities[f"{spec.role}:{spec.format}"] = {**identity, "prompts_sha256": manifest["prompts_sha256"],
                                                    "complete_prefix_items": manifest["complete_prefix_items"],
                                                    "sessions": manifest["sessions"]}
    for role in ROLES:
        rows = [value for key, value in identities.items() if key.startswith(role + ":")]
        for field in ("server_flags", "server_binaries_sha256", "output_tokens", "parallel_requests"):
            if len({json.dumps(row[field], sort_keys=True) for row in rows}) > 1:
                raise CalibrationError(f"{role} runs differ in {field}")
    return identities


def size(args: argparse.Namespace) -> dict[str, Any]:
    """Blinded interim: discordant totals of the first PILOT_ITEMS items per model and comparison, and the rule's N.
    Accuracies, gains, losses and signs are neither computed into the output nor printed."""
    items = load_items(args.gsm8k, "train")
    sequences = calibration_sequences(items)
    runs = _load_all(args.run, sequences, {role: PILOT_ITEMS for role in ROLES})
    result: dict[str, Any] = {"schema": SIZE_SCHEMA, "calibration_id": CALIBRATION_ID, "pilot_items": PILOT_ITEMS,
                              "rule": {"half_width": TARGET_HALF_WIDTH, "confidence": CONFIDENCE,
                                       "minimum": PILOT_ITEMS, "maximum": MAXIMUM_ITEMS, "rounding": ITEMS_ROUNDING},
                              "models": {}}
    for role in ROLES:
        item_ids = [item.item_id for item in sequences[role][:PILOT_ITEMS]]
        discordance = {}
        for name, base, treat in COMPARISONS:
            if (role, base) not in runs or (role, treat) not in runs:
                raise CalibrationError(f"sizing needs {role}:{base} and {role}:{treat}")
            discordant = sum(runs[(role, base)][item_id].correct != runs[(role, treat)][item_id].correct
                             for item_id in item_ids)
            discordance[name] = {"discordant": discordant, "pairs": len(item_ids),
                                 "discordance": discordant / len(item_ids)}
        psi_max = max(row["discordance"] for row in discordance.values())
        n = required_items(psi_max)
        result["models"][role] = {"discordance": discordance, "planning_discordance": psi_max, "items": n,
                                  "expected_half_width_at_items": expected_half_width(psi_max, n)}
    return result


def _sum_tables(tables: list[dict[str, Any]]) -> PairedCounts:
    return PairedCounts(both=sum(t["both_correct"] for t in tables), loss=sum(t["loss"] for t in tables),
                        gain=sum(t["gain"] for t in tables), neither=sum(t["neither_correct"] for t in tables))


def score(args: argparse.Namespace) -> dict[str, Any]:
    items = load_items(args.gsm8k, "train")
    sequences = calibration_sequences(items)
    limit = {role: getattr(args, f"{role}_items") for role in ROLES}
    identities = _check_identities(args.run)
    for spec in args.run:
        prefix = identities[f"{spec.role}:{spec.format}"]["complete_prefix_items"]
        if prefix < limit[spec.role]:
            raise CalibrationError(f"{spec.role}:{spec.format} completed {prefix} items, fewer than "
                                   f"{limit[spec.role]}")
    prompt_hashes = {}
    for role in ROLES:
        hashes = {identities[key]["prompts_sha256"] for key in identities if key.startswith(role + ":")}
        if len(hashes) > 1:
            raise CalibrationError(f"{role} runs used different prompt files")
        prompt_hashes[role] = hashes.pop() if hashes else None
    pieces = {}
    for value in args.tokenizer_model:
        role, _, path = value.partition("=")
        if role not in ROLES or not path:
            raise CalibrationError("--tokenizer-model expects ROLE=GGUF")
        pieces[role] = PieceTable.from_gguf(Path(path))
    runs = _load_all(args.run, sequences, limit, pieces)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA, "calibration_id": CALIBRATION_ID, "extraction_rule": protocol.EXTRACTION_ID,
        "output_tokens": OUTPUT_TOKENS, "confidence": CONFIDENCE, "items": limit,
        "item_lists_sha256": {role: item_list_sha256(sequences[role][:limit[role]]) for role in ROLES},
        "prompts_sha256": prompt_hashes, "runs": identities, "formats": {}, "comparisons": {}, "pooled": {}}
    for role in ROLES:
        item_ids = [item.item_id for item in sequences[role][:limit[role]]]
        report["formats"][role] = {fmt: format_summary([runs[(role, fmt)][item_id] for item_id in item_ids])
                                   for fmt in FORMATS if (role, fmt) in runs}
        report["comparisons"][role] = {}
        for name, base, treat in (*COMPARISONS, *DESCRIPTIVE_COMPARISONS):
            if (role, base) not in runs or (role, treat) not in runs:
                continue
            counts = paired(runs[(role, base)], runs[(role, treat)], item_ids)
            tolerant = paired(runs[(role, base)], runs[(role, treat)], item_ids, "tolerant")
            report["comparisons"][role][name] = {
                "baseline": base, "treatment": treat, **comparison_summary(counts),
                "similarity": similarity_counts(runs[(role, base)], runs[(role, treat)], item_ids),
                "tolerant_sensitivity": comparison_summary(tolerant)}
    for name, base, treat in (*COMPARISONS, *DESCRIPTIVE_COMPARISONS):
        tables = [report["comparisons"][role].get(name) for role in ROLES]
        if not all(tables):
            continue
        report["pooled"][name] = {"baseline": base, "treatment": treat, **comparison_summary(_sum_tables(tables)),
                                  "tolerant_sensitivity": comparison_summary(_sum_tables(
                                      [table["tolerant_sensitivity"] for table in tables]))}
    report["planning"] = {name: planning(row["discordance"]) for name, row in report["pooled"].items()}
    report["pooled_formats"] = {}
    for fmt in FORMATS:
        rows = [report["formats"][role].get(fmt) for role in ROLES]
        if all(rows):
            correct, n = sum(row["correct"] for row in rows), sum(row["n"] for row in rows)
            report["pooled_formats"][fmt] = {"n": n, "correct": correct, "accuracy": correct / n,
                                             "accuracy_wilson_95": list(_wilson(correct, n))}
    return report


def markdown(report: dict[str, Any]) -> str:
    pct = lambda value: f"{100 * value:.2f}"  # noqa: E731
    lines = [f"Items: Qwen {report['items']['qwen']}, Gemma {report['items']['gemma']} (GSM8K train, disjoint); "
             f"budget {report['output_tokens']} tokens; rule `{report['extraction_rule']}`.", "",
             "| model | format | n | accuracy % | Wilson 95 % | extracted % | 'Final answer' % | no answer in 512 | "
             "float near-misses | positioned | misses @256/320/384 of positioned | "
             "answer ends (p50 / p90 / max tokens) |",
             "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for role in ROLES:
        for fmt, row in report["formats"][role].items():
            ends = row["answer_token_end"] or {}
            misses = row["misses_by_budget_of_positioned"]
            lines.append(f"| {role} | {fmt} | {row['n']} | {pct(row['accuracy'])} | "
                         f"[{pct(row['accuracy_wilson_95'][0])}, {pct(row['accuracy_wilson_95'][1])}] | "
                         f"{pct(row['extraction_rate'])} | {pct(row['final_answer_rate'])} | "
                         f"{row['no_answer_within_budget']} | {row['float_near_misses']} | "
                         f"{row['positions_available']} | "
                         f"{misses['256']} / {misses['320']} / {misses['384']} | "
                         f"{ends.get('p50', '-')} / {ends.get('p90', '-')} / {ends.get('max', '-')} |")
    for fmt, row in report.get("pooled_formats", {}).items():
        lines.append(f"| pooled | {fmt} | {row['n']} | {pct(row['accuracy'])} | "
                     f"[{pct(row['accuracy_wilson_95'][0])}, {pct(row['accuracy_wilson_95'][1])}] | | | | | | | |")
    lines += ["", "| comparison (treatment - baseline) | model | n | gains | losses | discordance % | d (points) | "
              "Tango 95 % | Newcombe 95 % | McNemar p | identical outputs % | same answer % |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |"]
    for name, _, _ in (*COMPARISONS, *DESCRIPTIVE_COMPARISONS):
        for role in (*ROLES, "pooled"):
            row = (report["pooled"] if role == "pooled" else report["comparisons"][role]).get(name)
            if row is None:
                continue
            sim = row.get("similarity")
            lines.append(
                f"| {name}: {row['treatment']} - {row['baseline']} | {role} | {row['n']} | {row['gain']} | "
                f"{row['loss']} | {pct(row['discordance'])} | {100 * row['difference']:+.2f} | "
                f"[{100 * row['tango_95'][0]:+.2f}, {100 * row['tango_95'][1]:+.2f}] | "
                f"[{100 * row['newcombe_95'][0]:+.2f}, {100 * row['newcombe_95'][1]:+.2f}] | "
                f"{row['exact_mcnemar_p_two_sided']:.3f} | "
                f"{pct(sim['identical_share']) if sim else '-'} | {pct(sim['same_answer_share']) if sim else '-'} |")
    lines += ["", f"Sensitivity (descriptive, not the frozen rule): an answer within {TOLERANCE:g} relative of the "
              "reference counts as right (floating-point formatting such as 46.00000000000001).", "",
              "| comparison | model | gains | losses | d (points) | Tango 95 % |",
              "| --- | --- | ---: | ---: | ---: | --- |"]
    for name, _, _ in (*COMPARISONS, *DESCRIPTIVE_COMPARISONS):
        for role in (*ROLES, "pooled"):
            row = (report["pooled"] if role == "pooled" else report["comparisons"][role]).get(name)
            if row is None:
                continue
            row = row["tolerant_sensitivity"]
            lines.append(f"| {name} | {role} | {row['gain']} | {row['loss']} | {100 * row['difference']:+.2f} | "
                         f"[{100 * row['tango_95'][0]:+.2f}, {100 * row['tango_95'][1]:+.2f}] |")
    if report.get("planning"):
        margins = [f"{margin:.3f}" for margin in PLANNING_MARGINS]
        lines += ["", "Planning (pooled discordance; true d = 0, one-sided alpha 0.025): pairs for power 0.8 "
                  "(asymptotic Tango) / exact power at " + " and ".join(str(p) for p in PLANNING_PAIRS) + " pairs.", "",
                  "| discordance source | psi % | " + " | ".join(f"M = {100 * float(m):g} pts" for m in margins) + " |",
                  "| --- | ---: | " + " | ".join("---" for _ in margins) + " |"]
        for name, row in report["planning"].items():
            cells = []
            for margin in margins:
                powers = " / ".join(f"{row['exact_power'][str(p)][margin]:.2f}" for p in PLANNING_PAIRS)
                cells.append(f"{row['pairs_for_power_asymptotic'][margin]} ({powers})")
            lines.append(f"| {name} | {pct(row['discordance'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("run")
    sub.add_argument("--gsm8k", type=Path, required=True, help="pinned GSM8K train jsonl")
    sub.add_argument("--role", choices=ROLES, required=True)
    sub.add_argument("--format", choices=FORMATS, required=True)
    sub.add_argument("--model", type=Path, required=True)
    sub.add_argument("--model-sha256", default=None, help="skip hashing the model (recorded as given)")
    sub.add_argument("--verify-model-sha256", action="store_true")
    sub.add_argument("--server", type=Path, required=True, help="llama-server binary")
    sub.add_argument("--gpu", required=True, help="nvidia-smi index; exported as CUDA_VISIBLE_DEVICES")
    sub.add_argument("--port", type=int, required=True)
    sub.add_argument("--items", type=int, required=True, help="first N items of the model's sequence")
    sub.add_argument("--prompts", type=Path, required=True, help="reference prompt tokens (jsonl) of this model")
    sub.add_argument("--write-prompts", action="store_true",
                     help="create / extend the reference prompts from this run's tokenizer (the q4 file)")
    sub.add_argument("--allow-tokenizer-mismatch", action="store_true")
    sub.add_argument("--allow-busy-gpu", action="store_true")
    sub.add_argument("--run-dir", type=Path, required=True)
    for name in ("size", "score"):
        sub = commands.add_parser(name)
        sub.add_argument("--gsm8k", type=Path, required=True)
        sub.add_argument("--run", type=parse_run, action="append", required=True, metavar="ROLE:FORMAT=DIR")
        sub.add_argument("--out", type=Path, required=True)
        if name == "score":
            sub.add_argument("--tokenizer-model", action="append", default=[], metavar="ROLE=GGUF",
                             help="vocabulary for per-token offsets (the q4 file of the model)")
            sub.add_argument("--qwen-items", type=int, required=True)
            sub.add_argument("--gemma-items", type=int, required=True)
            sub.add_argument("--md", type=Path)
    args = parser.parse_args(argv)
    value = {"run": run, "size": size, "score": score}[args.command](args)
    if args.command in ("size", "score"):
        args.out.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="ascii")
        if args.command == "score" and args.md is not None:
            args.md.write_text(markdown(value), encoding="ascii")
        if args.command == "size":
            print(json.dumps({role: {"items": row["items"], "discordance": {
                name: entry["discordance"] for name, entry in row["discordance"].items()}}
                for role, row in value["models"].items()}, sort_keys=True))
        else:
            print(json.dumps({"out": str(args.out)}))
    else:
        print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
