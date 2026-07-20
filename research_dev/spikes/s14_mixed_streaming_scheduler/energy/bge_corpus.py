#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


WORDBANK = (
    "machine learning models encode natural language text into dense vector "
    "representations capturing semantic meaning for retrieval reranking clustering "
    "and classification across many downstream tasks efficiently and reliably at scale "
    "using transformer encoders with attention layers normalization and feed forward "
    "networks trained on large corpora of documents queries and passages"
).split()


def make_prompt(target_words: int) -> str:
    if isinstance(target_words, bool) or not isinstance(target_words, int) or target_words <= 0:
        raise ValueError("target_words must be a positive integer")
    return " ".join((WORDBANK * ((target_words // len(WORDBANK)) + 1))[:target_words])


def write_corpus(path: Path, prompt: str, n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    path.write_text("\n".join([prompt] * n) + "\n")
