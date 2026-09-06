"""
Illustration of the two-stage decoupled architecture from Section 3 of:
"Reinforcement Learning with Metacognitive Feedback Elicits Faithful
Uncertainty Expression in LLMs" (arXiv:2606.32032).

Stage 1 (heavy, RL-trained):  produces a scalar confidence in [0, 1].
Stage 2 (light, deterministic): maps that confidence + a chosen style
                                to a natural-language phrase.

The point: the RL policy only learns to emit faithful NUMERICAL confidence.
How that number is verbalized is a separate, pluggable component — so you
can retarget the linguistic register (legal, casual, formal, ...) without
retraining RL.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol


# ---------------------------------------------------------------------------
# Stage 1 — RL-trained "numerical confidence" head (stand-in)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelState:
    """A minimal stand-in for the hidden state of the LLM at decision time."""
    logits_for_correct: float   # logit favoring the correct answer
    logits_for_incorrect: float # logit favoring any wrong answer


class NumericalConfidenceModel(Protocol):
    """Anything that turns a (query, state) pair into a confidence in [0, 1]."""
    def predict_confidence(self, query: str, state: ModelState) -> float: ...


class ToyRLConfidenceHead:
    """
    Mimics the role of the RL-trained head: maps the model's internal signal
    (here: logit gap) into a calibrated scalar confidence via a learned
    logistic. In the real paper this head is trained with the metacognitive
    reward; here we just pick a plausible sigmoid.

    NOTE: this class is a stand-in. Replace with the actual policy's head
    once the RL stage has been run.
    """

    def __init__(self, temperature: float = 1.5) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self._temperature = temperature

    def predict_confidence(self, query: str, state: ModelState) -> float:
        gap = state.logits_for_correct - state.logits_for_incorrect
        # Same shape as a softmax-derived confidence, modulated by temperature.
        confidence = 1.0 / (1.0 + math.exp(-gap / self._temperature))
        # Numerical safety: clip away from exact 0/1 to keep verbalizers sane.
        return max(min(confidence, 1.0 - 1e-6), 1e-6)


# ---------------------------------------------------------------------------
# Stage 2 — deterministic, style-aware verbalizer (the "light" stage)
# ---------------------------------------------------------------------------

# Each style is just a deterministic function: float -> str.
ConfidenceToPhrase = Callable[[float], str]


def _legal_style(c: float) -> str:
    if c >= 0.90:
        return "I hold the opinion, with a high degree of professional certainty, that the answer is correct."
    if c >= 0.70:
        return "It is my considered professional view that the answer is likely correct, subject to ordinary review."
    if c >= 0.50:
        return "On balance, the answer appears more probable than not, though contrary authority is reasonably available."
    if c >= 0.30:
        return "I am unable to assert the answer with confidence; the contrary position is, in my assessment, substantive."
    return "I do not profess confidence in the answer; the matter is, in my professional judgment, genuinely uncertain."


def _casual_style(c: float) -> str:
    if c >= 0.90: return "Yeah, pretty sure that's right."
    if c >= 0.70: return "I think it's correct, but I'd double-check."
    if c >= 0.50: return "Maybe — I'm honestly not sure."
    if c >= 0.30: return "I'd guess no, but I really don't know."
    return "I have no idea, honestly."


def _formal_style(c: float) -> str:
    if c >= 0.90: return "The answer is, with high confidence, correct."
    if c >= 0.70: return "The answer is likely correct."
    if c >= 0.50: return "The answer is plausible but not established."
    if c >= 0.30: return "The answer is doubtful."
    return "The answer cannot be determined with any reasonable confidence."


# Pluggable registry: add new styles here without touching Stage 1.
STYLE_REGISTRY: Mapping[str, ConfidenceToPhrase] = {
    "legal":  _legal_style,
    "casual": _casual_style,
    "formal": _formal_style,
}


class StylisticRewriter:
    """The 'light' stage: a thin layer that turns numbers into words."""

    def __init__(self, styles: Mapping[str, ConfidenceToPhrase] = STYLE_REGISTRY) -> None:
        self._styles = dict(styles)

    @property
    def available_styles(self) -> tuple[str, ...]:
        return tuple(self._styles)

    def register_style(self, name: str, verbalizer: ConfidenceToPhrase) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("style name must be a non-empty string")
        self._styles[name] = verbalizer

    def render(self, confidence: float, style: str) -> str:
        if style not in self._styles:
            raise KeyError(
                f"unknown style {style!r}; available: {self.available_styles}"
            )
        verbalizer = self._styles[style]
        try:
            phrase = verbalizer(confidence)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"verbalizer for style {style!r} failed") from exc
        return phrase


# ---------------------------------------------------------------------------
# Pipeline — wires the two stages together
# ---------------------------------------------------------------------------

class UncertaintyPipeline:
    """
    Compose Stage 1 (RL confidence) with Stage 2 (stylistic rewriting).

    Because the stages are decoupled you can:
      * swap the RL head without touching the verbalizers;
      * add / change styles without ever retraining RL.
    """

    def __init__(
        self,
        confidence_model: NumericalConfidenceModel,
        rewriter: StylisticRewriter | None = None,
    ) -> None:
        self._model = confidence_model
        self._rewriter = rewriter or StylisticRewriter()

    def answer(
        self,
        query: str,
        state: ModelState,
        style: str = "formal",
        leading_answer: str | None = None,
    ) -> str:
        c = self._model.predict_confidence(query, state)
        hedge = self._rewriter.render(c, style)
        return f"{leading_answer + ' ' if leading_answer else ''}{hedge} [c={c:.2f}]"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    rl_head = ToyRLConfidenceHead(temperature=1.5)
    rewriter = StylisticRewriter()
    # Add a brand-new style on the fly — no retraining required.
    rewriter.register_style(
        "hedge",
        lambda c: "Perhaps." if c > 0.5 else "I really cannot say.",
    )
    pipe = UncertaintyPipeline(confidence_model=rl_head, rewriter=rewriter)

    # Synthesize a few internal states to drive the demo.
    rng = random.Random(0)
    print("Same RL head, different styles — illustrating the decoupling:\n")
    for i in range(5):
        state = ModelState(
            logits_for_correct=rng.uniform(-2, 4),
            logits_for_incorrect=rng.uniform(-2, 2),
        )
        print(f"--- example {i + 1} (logit gap = "
              f"{state.logits_for_correct - state.logits_for_incorrect:+.2f}) ---")
        for style in ("legal", "casual", "formal", "hedge"):
            print(f"  [{style:>6}] {pipe.answer('Q', state, style=style)}")
        print()


if __name__ == "__main__":
    _demo()