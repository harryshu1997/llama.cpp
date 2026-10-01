"""GSM8K items, per-model chat prompts and the frozen final-answer extraction rule."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .protocol import GSM8K_ROWS, GSM8K_SHA256


class QualityDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class Gsm8kItem:
    split: str
    index: int
    question: str
    reference: Decimal

    @property
    def item_id(self) -> str:
        return f"gsm8k-{self.split}:{self.index:04d}"


def parse_reference(answer: str) -> Decimal:
    """The number after the final '####' of a GSM8K solution (commas removed)."""
    if type(answer) is not str or "####" not in answer:
        raise QualityDataError("GSM8K solution has no final answer")
    value = parse_number(answer.rsplit("####", 1)[1].strip())
    if value is None:
        raise QualityDataError("GSM8K final answer is not a number")
    return value


def load_items(path: Path, split: str, *, verify: bool = True) -> list[Gsm8kItem]:
    data = path.read_bytes()
    if verify:
        if split not in GSM8K_SHA256:
            raise QualityDataError(f"unknown GSM8K split {split}")
        observed = hashlib.sha256(data).hexdigest()
        if observed != GSM8K_SHA256[split]:
            raise QualityDataError(f"GSM8K {split} sha256 {observed} differs from the pinned copy")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if verify and len(rows) != GSM8K_ROWS[split]:
        raise QualityDataError("GSM8K row count differs from the pinned copy")
    items = []
    for index, row in enumerate(rows):
        if type(row) is not dict or type(row.get("question")) is not str:
            raise QualityDataError(f"GSM8K row {index} is malformed")
        items.append(Gsm8kItem(split=split, index=index, question=row["question"].strip(),
                               reference=parse_reference(row.get("answer"))))
    return items


USER_TEMPLATE = (
    "Solve the following math word problem. Reason step by step, then write the final answer on its own "
    "line in the form \"Final answer: <number>\".\n\nProblem: {question}"
)

# Official chat-template renderings with add_generation_prompt. The codec adds each tokenizer's BOS
# (Gemma, Llama) itself, so no BOS text appears here. Qwen3 uses its non-thinking rendering
# (enable_thinking=false: an empty think block); Gemma 4 its default non-thinking rendering (empty thought
# channel); Llama 3.2 its default system header with the template's fixed date.
PROMPT_TEMPLATES = {
    "qwen": "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "gemma": "<|turn>user\n{content}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
    "llama": ("<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n"
              "Today Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{content}"
              "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
}

# Token structure the codec must produce for each rendering (ids of the pinned tokenizers; the Gemma and
# Llama ids are the ones in the frozen longtail_eval_v2 prompts).
EXPECTED_FIRST_TOKENS = {"qwen": (151644,), "gemma": (2, 105), "llama": (128000, 128006)}
EXPECTED_LAST_TOKENS = {
    "qwen": (151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271),
    "gemma": (106, 107, 105, 4368, 107, 100, 45518, 107, 101),
    "llama": (128009, 128006, 78191, 128007, 271),
}


def render_prompt(role: str, question: str) -> str:
    if role not in PROMPT_TEMPLATES:
        raise QualityDataError(f"no prompt template for {role}")
    return PROMPT_TEMPLATES[role].format(content=USER_TEMPLATE.format(question=question.strip()))


_NUMBER = re.compile(r"(?P<sign>[-\u2212])?(?:\$\s?)?(?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")
_FINAL = re.compile(r"final\s*answer\s*(?:\*\*|__)?\s*(?::|\uff1a|\bis\b|=)", re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\s*\{")
_ANSWER_IS = re.compile(r"\banswer\s+is\s*:?", re.IGNORECASE)
_FINAL_WINDOW = 80
_BOXED_WINDOW = 40
_ANSWER_IS_WINDOW = 40


def parse_number(text: str) -> Decimal | None:
    match = _NUMBER.match(text.strip())
    if match is None:
        return None
    return _decimal(match)


def _decimal(match: re.Match) -> Decimal | None:
    try:
        value = Decimal(match.group("digits").replace(",", ""))
    except InvalidOperation:
        return None
    return -value if match.group("sign") else value


@dataclass(frozen=True)
class Extraction:
    value: Decimal | None
    rule: str
    start: int
    end: int

    @property
    def found(self) -> bool:
        return self.value is not None

    def to_json(self) -> dict[str, object]:
        return {"value": None if self.value is None else format(self.value.normalize(), "f"),
                "rule": self.rule, "start": self.start, "end": self.end}


def _first_number(text: str, marker: re.Pattern, window: int, rule: str) -> Extraction | None:
    for found in marker.finditer(text):
        number = _NUMBER.search(text, found.end(), found.end() + window)
        if number is not None:
            value = _decimal(number)
            if value is not None:
                return Extraction(value, rule, found.start(), number.end())
    return None


def extract_answer(text: str) -> Extraction:
    """Frozen rule gsm8k-first-final-answer-v1.

    Generation runs a fixed token budget with end-of-generation tokens banned, so text after the model's
    first conclusion is an arbitrary continuation: the rule therefore takes the EARLIEST conclusion, never
    the last number. A conclusion is 'final answer' followed by ':' / 'is' / '=' (markdown emphasis
    allowed) with a number within 80 characters, or a LaTeX \\boxed{...} holding a number within 40
    characters, whichever starts first. Only if neither exists, the first 'answer is' followed by a number
    within 40 characters. Otherwise the output is unscorable (counted as incorrect)."""
    candidates = [row for row in (_first_number(text, _FINAL, _FINAL_WINDOW, "final-answer"),
                                  _first_number(text, _BOXED, _BOXED_WINDOW, "boxed")) if row is not None]
    if candidates:
        return min(candidates, key=lambda row: row.start)
    fallback = _first_number(text, _ANSWER_IS, _ANSWER_IS_WINDOW, "answer-is")
    return fallback if fallback is not None else Extraction(None, "none", -1, -1)


def is_correct(extraction: Extraction, reference: Decimal) -> bool:
    return extraction.value is not None and extraction.value == reference

