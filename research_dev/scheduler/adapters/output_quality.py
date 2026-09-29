"""Conservative semantic-sanity checks for physical text output."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticOutputAssessment:
    accepted: bool
    reasons: tuple[str, ...]
    character_count: int
    lexical_character_count: int
    token_count: int
    unique_token_count: int
    mode: str = "semantic-sanity-v1"

    def to_json(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "character_count": self.character_count,
            "lexical_character_count": self.lexical_character_count,
            "mode": self.mode,
            "reasons": list(self.reasons),
            "token_count": self.token_count,
            "unique_token_count": self.unique_token_count,
        }


def assess_semantic_output(
    text: str, tokens: tuple[int, ...]
) -> SemanticOutputAssessment:
    """Reject malformed or clearly degenerate output without exact matching."""
    if type(text) is not str or any(type(token) is not int for token in tokens):
        raise ValueError("semantic output input is invalid")
    reasons = []
    stripped = text.strip()
    controls = sum(
        1 for character in text
        if not character.isprintable() and not character.isspace()
    )
    lexical = sum(character.isalnum() for character in text)
    if not stripped:
        reasons.append("EMPTY_TEXT")
    if controls and controls * 50 > max(1, len(text)):
        reasons.append("CONTROL_CHARACTER_DENSITY")
    if stripped and lexical == 0:
        reasons.append("NO_LEXICAL_CONTENT")
    if len(tokens) >= 16:
        counts = Counter(tokens)
        if max(counts.values()) * 20 >= len(tokens) * 19:
            reasons.append("DEGENERATE_TOKEN_REPETITION")
        else:
            for period in range(1, min(4, len(tokens) // 4) + 1):
                mismatches = sum(
                    token != tokens[index % period]
                    for index, token in enumerate(tokens)
                )
                if mismatches * 20 <= len(tokens):
                    reasons.append("DEGENERATE_TOKEN_CYCLE")
                    break
    return SemanticOutputAssessment(
        accepted=not reasons,
        reasons=tuple(reasons),
        character_count=len(text),
        lexical_character_count=lexical,
        token_count=len(tokens),
        unique_token_count=len(set(tokens)),
    )


def accounting_output_assessment(
    text: str, tokens: tuple[int, ...]
) -> SemanticOutputAssessment:
    """Describe internal warm-up output without applying a quality gate."""
    if type(text) is not str or any(type(token) is not int for token in tokens):
        raise ValueError("accounting output input is invalid")
    return SemanticOutputAssessment(
        accepted=True,
        reasons=(),
        character_count=len(text),
        lexical_character_count=sum(character.isalnum() for character in text),
        token_count=len(tokens),
        unique_token_count=len(set(tokens)),
        mode="accounting-only-v1",
    )
