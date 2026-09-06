"""
Rewriting Protocol (Sec. 3.4 of arXiv:2606.32032 — RLMF).
Convierte scores numéricos de confianza calibrados en hedges lingüísticos.

Diseño:
- Scores en [0, 1] ya vienen del paso de RL (calibración aplicada).
- Se discretizan en 5 niveles y se proyectan a muletillas contextuales.
- El dominio y el tipo de claim modulan el hedge final.

Ejecutar como script: `python rewriting_protocol.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Final


# ---------------------------------------------------------------------------
# 1. Niveles de confianza discretizados
# ---------------------------------------------------------------------------
class ConfidenceLevel(Enum):
    """Discretización del score continuo en categorías lingüísticas."""

    VERY_LOW = "very_low"        # 0.00 – 0.20  → "muy poco probable"
    LOW = "low"                  # 0.20 – 0.40  → "probablemente no"
    UNCERTAIN = "uncertain"      # 0.40 – 0.60  → "no está claro / quizás"
    HIGH = "high"                # 0.60 – 0.80  → "probablemente sí"
    VERY_HIGH = "very_high"      # 0.80 – 1.00  → "con alta confianza"


_THRESHOLDS: Final[tuple[float, ...]] = (0.20, 0.40, 0.60, 0.80)


def score_to_level(score: float) -> ConfidenceLevel:
    """Mapea un score continuo [0, 1] a un ConfidenceLevel."""
    if math.isnan(score) or math.isinf(score):
        raise ValueError(f"Score inválido: {score!r}")
    s = max(0.0, min(1.0, score))  # clipping defensivo
    if s < _THRESHOLDS[0]:
        return ConfidenceLevel.VERY_LOW
    if s < _THRESHOLDS[1]:
        return ConfidenceLevel.LOW
    if s < _THRESHOLDS[2]:
        return ConfidenceLevel.UNCERTAIN
    if s < _THRESHOLDS[3]:
        return ConfidenceLevel.HIGH
    return ConfidenceLevel.VERY_HIGH


# ---------------------------------------------------------------------------
# 2. Tabla de hedges por (nivel, dominio, claim_type)
# ---------------------------------------------------------------------------
# claim_type ∈ {"factual", "prognostic"}: los hechos admiten hedges más
# fuertes que las predicciones (alineado con la Sec. 3.4 del paper).

HEDGES: Final[dict[tuple[ConfidenceLevel, str], str]] = {
    # ---- Dominio: médico ----
    (ConfidenceLevel.VERY_HIGH, "medical"):    "con alta certeza clínica",
    (ConfidenceLevel.HIGH, "medical"):        "lo más probable es que",
    (ConfidenceLevel.UNCERTAIN, "medical"):   "no es posible descartar",
    (ConfidenceLevel.LOW, "medical"):          "existe evidencia preliminar en contra de",
    (ConfidenceLevel.VERY_LOW, "medical"):     "no hay indicios que apoyen",

    # ---- Dominio: factual general ----
    (ConfidenceLevel.VERY_HIGH, "factual"):    "de manera consolidada",
    (ConfidenceLevel.HIGH, "factual"):        "con base en la evidencia disponible",
    (ConfidenceLevel.UNCERTAIN, "factual"):   "existen posturas divergentes sobre",
    (ConfidenceLevel.LOW, "factual"):          "hay argumentos en contra de",
    (ConfidenceLevel.VERY_LOW, "factual"):     "no se sostiene que",

    # ---- Dominio: legal ----
    (ConfidenceLevel.VERY_HIGH, "legal"):      "con fundamento jurídico sólido",
    (ConfidenceLevel.HIGH, "legal"):          "es razonable sostener que",
    (ConfidenceLevel.UNCERTAIN, "legal"):     "el criterio no es uniforme respecto a",
    (ConfidenceLevel.LOW, "legal"):            "resulta cuestionable afirmar que",
    (ConfidenceLevel.VERY_LOW, "legal"):       "carece de respaldo normativo que",
}

DOMAIN_FALLBACK: Final[dict[ConfidenceLevel, str]] = {
    ConfidenceLevel.VERY_HIGH: "con alta confianza",
    ConfidenceLevel.HIGH:       "probablemente",
    ConfidenceLevel.UNCERTAIN:  "no es seguro que",
    ConfidenceLevel.LOW:        "poco probable que",
    ConfidenceLevel.VERY_LOW:   "muy improbable que",
}

# Modificadores para predicciones (prognostic) — más cautos.
PROGNOSTIC_MODIFIER: Final[str] = "según la trayectoria esperada,"


# ---------------------------------------------------------------------------
# 3. Protocolo de reescritura
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RewrittenClaim:
    """Salida del Rewriting Protocol."""

    claim: str
    score: float
    level: ConfidenceLevel
    hedge: str
    domain: str
    claim_type: str

    def render(self) -> str:
        """Devuelve la frase final con el hedge antepuesto."""
        prefix = (
            f"{PROGNOSTIC_MODIFIER} {self.hedge}"
            if self.claim_type == "prognostic" and self.level
            in {ConfidenceLevel.UNCERTAIN, ConfidenceLevel.LOW,
                ConfidenceLevel.VERY_LOW}
            else self.hedge
        )
        return f"{prefix} {self.claim}".strip().capitalize()


class RewritingProtocol:
    """Aplica el mapeo score → hedge lingüístico."""

    def __init__(
        self,
        hedges: dict[tuple[ConfidenceLevel, str], str] | None = None,
    ) -> None:
        self._hedges = hedges if hedges is not None else HEDGES

    def rewrite(
        self,
        claim: str,
        score: float,
        *,
        domain: str = "factual",
        claim_type: str = "factual",
    ) -> RewrittenClaim:
        """Convierte un claim + score calibrado en una versión hedged."""
        level = score_to_level(score)
        hedge = self._hedges.get(
            (level, domain),
            DOMAIN_FALLBACK[level],
        )
        return RewrittenClaim(
            claim=claim,
            score=score,
            level=level,
            hedge=hedge,
            domain=domain,
            claim_type=claim_type,
        )


# ---------------------------------------------------------------------------
# 4. Demo + sanity checks
# ---------------------------------------------------------------------------
def _demo() -> None:
    protocol = RewritingProtocol()
    examples: list[tuple[str, float, str, str]] = [
        ("el paciente desarrollará sepsis en 48 h", 0.85, "medical",   "prognostic"),
        ("la PCR es positiva",                       0.92, "medical",   "factual"),
        ("el contrato es nulo por vicios",           0.55, "legal",     "factual"),
        ("España limita el acceso a datos biométricos de los pacientes", 0.30, "legal", "factual"),
        ("la mitosis produce células diploides",     0.18, "factual",   "factual"),
    ]
    for claim, score, dom, ctype in examples:
        out = protocol.rewrite(claim, score, domain=dom, claim_type=ctype)
        print(f"[score={score:.2f} | {out.level.value}] {out.render()}")


def _tests() -> None:
    p = RewritingProtocol()
    out = p.rewrite("afirmación X", 0.9, domain="factual")
    assert out.level is ConfidenceLevel.VERY_HIGH
    assert out.render().startswith("De manera consolidada")

    out = p.rewrite("afirmación X", 0.15, domain="medical")
    assert out.level is ConfidenceLevel.VERY_LOW
    assert "no hay indicios" in out.render()

    # clipping defensivo
    out = p.rewrite("X", 1.5, domain="factual")
    assert out.level is ConfidenceLevel.VERY_HIGH

    # error con NaN
    try:
        p.rewrite("X", math.nan, domain="factual")
    except ValueError:
        pass
    else:
        raise AssertionError("Se esperaba ValueError con NaN")

    print("✔ tests OK")


if __name__ == "__main__":
    _tests()
    _demo()