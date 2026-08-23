"""
Parse an LLM output (answer + per-sentence confidence) into structured
(sentence, confidence) pairs, suitable for downstream calibration metrics
such as ECE or Brier score at the sentence level.

Assumes the model was trained to emit blocks like:
    <answer>I am fairly sure. The second claim is uncertain.</answer>
    <confidence>0.9,0.4</confidence>
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# A minimal regex-based sentence splitter that tolerates common abbreviations
# (e.g., "e.g.", "i.e.", "Mr.") and decimals. It is *not* perfect for every
# language, but it is good enough for English evaluation pipelines.
_SENTENCE_END = re.compile(
    r"""
    (?<=[.!?])            # punctuation that ends a sentence
    (?=                   # followed by
        \s+               # whitespace
        [A-Z"'\(\[]       # and a likely sentence-start char
    )
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class SentenceWithConfidence:
    index: int
    text: str
    confidence: float  # in [0.0, 1.0]


class OutputParseError(ValueError):
    """Raised when the model output does not match the expected schema."""


def split_into_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences while keeping the sentence content."""
    text = text.strip()
    if not text:
        return []
    # Split and strip whitespace; filter empties.
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def parse_model_output(raw: str) -> list[SentenceWithConfidence]:
    """
    Parse a model emission containing <answer>...</answer> and
    <confidence>...</confidence> tags into per-sentence records.

    Raises:
        OutputParseError: if tags are missing, counts mismatch, or a
            confidence value is not a float in [0, 1].
    """
    answer_match = re.search(r"<answer>(.*?)</answer>", raw, flags=re.DOTALL)
    conf_match = re.search(r"<confidence>(.*?)</confidence>", raw, flags=re.DOTALL)

    if not answer_match or not conf_match:
        raise OutputParseError("Missing <answer> or <confidence> block.")

    answer_text = answer_match.group(1).strip()
    conf_text = conf_match.group(1).strip()

    try:
        confidences = [float(x) for x in conf_text.split(",")]
    except ValueError as exc:
        raise OutputParseError(f"Non-numeric confidence value: {exc}") from exc

    if any(not 0.0 <= c <= 1.0 for c in confidences):
        raise OutputParseError("Confidence values must lie in [0, 1].")

    sentences = split_into_sentences(answer_text)

    if len(sentences) != len(confidences):
        raise OutputParseError(
            f"Mismatch: {len(sentences)} sentence(s) vs "
            f"{len(confidences)} confidence value(s)."
        )

    return [
        SentenceWithConfidence(index=i, text=s, confidence=c)
        for i, (s, c) in enumerate(zip(sentences, confidences))
    ]


if __name__ == "__main__":
    sample = (
        "<answer>The sky is blue due to Rayleigh scattering. "
        "I am less certain about the exact wavelength peak.</answer>"
        "<confidence>0.95,0.60</confidence>"
    )

    records = parse_model_output(sample)
    for r in records:
        print(f"[{r.index}] conf={r.confidence:.2f} | {r.text}")