"""
Metacognitive-Aware Sample Selection (MAS) — ejemplo ilustrativo para el artículo.

Intuición del paper §3.3: la misma señal metacognitiva que modula advantages
dentro del rollout (RLMF, nivel completion) puede usarse ANTES del RL para
filtrar qué muestras del corpus merecen entrar en el entrenamiento (MAS, nivel
muestra). Aquí se calcula una "ventaja de grupo" Z_g por muestra a partir de la
fidelidad con la que el modelo verbaliza su propia incertidumbre, y se aplica
una política de selección reproducible (umbral + razones de descarte).

Ejecutar: `python mas_sample_selection.py` (no descarga modelos).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Métricas de fidelidad verbalizada
# ---------------------------------------------------------------------------

def expected_calibration_error(
    p_verb: Sequence[float], correct: Sequence[int], n_bins: int = 10
) -> float:
    """ECE sobre probabilidades verbalizadas por el modelo.

    Cuanto más bajo, mejor: la confianza que el modelo "dice" se parece a su
    accuracy real.
    """
    p = np.asarray(p_verb, dtype=float)
    y = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p > lo) & (p <= hi)
        if not m.any():
            continue
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def brier(p_verb: Sequence[float], correct: Sequence[int]) -> float:
    """Brier score sobre probabilidades verbalizadas (menor = mejor)."""
    p = np.asarray(p_verb, dtype=float)
    y = np.asarray(correct, dtype=float)
    return float(np.mean((p - y) ** 2))


def metacognitive_reward(ece: float, brier: float) -> float:
    """Recompensa metacognitiva r_g ∈ [0, 1]; combina calibración y Brier."""
    # ECE y Brier ∈ [0, 1]; los pasamos a "fidelidad" y promediamos.
    return float(max(0.0, 1.0 - 0.5 * (ece + brier)))


# ---------------------------------------------------------------------------
# Datos sintéticos (sin red ni descargas)
# ---------------------------------------------------------------------------

@dataclass
class GroupCompletions:
    """Todas las completions del modelo para una misma pregunta/prompt."""
    sample_id: str
    p_verb: list[float]            # probabilidades verbalizadas
    correct: list[int]             # 1 si la completion acierta, 0 si no


@dataclass
class ScoredSample:
    sample_id: str
    p_verb_mean: float
    accuracy: float
    ece: float
    brier: float
    reward: float          # r_g  (fidelidad verbalizada del grupo)
    z_g: float             # ventaja por muestra, normalizada dentro del batch


# ---------------------------------------------------------------------------
# Núcleo MAS
# ---------------------------------------------------------------------------

def score_groups(groups: Iterable[GroupCompletions]) -> list[ScoredSample]:
    """Equivalente a 'computar ventaja' pero a nivel de muestra/grupo."""
    rows: list[ScoredSample] = []
    for g in groups:
        ece = expected_calibration_error(g.p_verb, g.correct)
        b = brier(g.p_verb, g.correct)
        r_g = metacognitive_reward(ece, b)
        rows.append(
            ScoredSample(
                sample_id=g.sample_id,
                p_verb_mean=float(np.mean(g.p_verb)),
                accuracy=float(np.mean(g.correct)),
                ece=ece,
                brier=b,
                reward=r_g,
                z_g=0.0,  # se rellena en el segundo pase
            )
        )
    # Z_g = (r_g - mean(r_g)) / std(r_g)  →  "ventaja de grupo"
    if len(rows) > 1:
        rs = np.array([r.reward for r in rows], dtype=float)
        mu, sd = float(rs.mean()), float(rs.std(ddof=0)) or 1e-9
        for r in rows:
            r.z_g = (r.reward - mu) / sd
    return rows


# Política de selección: qué muestras pasan el filtro metacognitivo
@dataclass
class SelectionConfig:
    z_g_min: float = -0.25     # descarta verbalizaciones muy desalineadas
    reward_min: float = 0.55   # piso absoluto de fidelidad
    keep_top_k: int | None = None  # alternativa: top-k por z_g


@dataclass
class SelectionDecision:
    sample_id: str
    keep: bool
    reason: str


def select(
    scored: Sequence[ScoredSample], cfg: SelectionConfig
) -> list[SelectionDecision]:
    decisions: list[SelectionDecision] = []
    if cfg.keep_top_k is not None:
        order = sorted(scored, key=lambda s: s.z_g, reverse=True)
        keep_ids = {s.sample_id for s in order[: cfg.keep_top_k]}
    else:
        keep_ids = None  # usar umbrales

    for s in scored:
        if keep_ids is not None:
            kept = s.sample_id in keep_ids
            reason = "top-k por Z_g" if kept else "fuera del top-k"
        else:
            kept = (s.z_g >= cfg.z_g_min) and (s.reward >= cfg.reward_min)
            reasons: list[str] = []
            if s.z_g < cfg.z_g_min:
                reasons.append(f"Z_g={s.z_g:+.2f}<{cfg.z_g_min:+.2f}")
            if s.reward < cfg.reward_min:
                reasons.append(f"r_g={s.reward:.2f}<{cfg.reward_min:.2f}")
            reason = "ok" if kept else "; ".join(reasons) or "umbral"
        decisions.append(SelectionDecision(s.sample_id, kept, reason))
    return decisions


# ---------------------------------------------------------------------------
# Demo reproducible (sin semillas aleatorias; datos deterministas)
# ---------------------------------------------------------------------------

def demo_corpus() -> list[GroupCompletions]:
    # 6 preguntas, 5 completions cada una, p_verb y correct sintéticos.
    raw = {
        "q1": ([0.90, 0.85, 0.92, 0.88, 0.91], [1, 1, 1, 1, 1]),  # bien calibrado
        "q2": ([0.30, 0.35, 0.25, 0.40, 0.28], [0, 0, 0, 0, 0]),  # bien calibrado
        "q3": ([0.95, 0.93, 0.96, 0.94, 0.92], [0, 1, 0, 0, 0]),  # sobreconfiado
        "q4": ([0.50, 0.48, 0.52, 0.51, 0.49], [1, 0, 1, 0, 1]),  # ok, ruido
        "q5": ([0.10, 0.15, 0.08, 0.12, 0.20], [0, 0, 0, 0, 0]),  # bien calibrado
        "q6": ([0.70, 0.75, 0.65, 0.78, 0.72], [0, 0, 1, 0, 0]),  # leve sobreconfianza
    }
    return [
        GroupCompletions(qid, list(p), list(c)) for qid, (p, c) in raw.items()
    ]


def main() -> None:
    groups = demo_corpus()
    scored = score_groups(groups)

    print(f"{'id':>4} {'acc':>5} {'ece':>6} {'brier':>6} {'r_g':>5} {'Z_g':>6}")
    for s in scored:
        print(f"{s.sample_id:>4} {s.accuracy:>5.2f} {s.ece:>6.3f} "
              f"{s.brier:>6.3f} {s.reward:>5.2f} {s.z_g:>+6.2f}")

    cfg = SelectionConfig(z_g_min=-0.25, reward_min=0.55)
    decisions = select(scored, cfg)

    kept = [d for d in decisions if d.keep]
    print(f"\nMAS → {len(kept)}/{len(decisions)} muestras pasan el filtro.")
    for d in decisions:
        marker = "✔" if d.keep else "✘"
        print(f"  {marker} {d.sample_id}: {d.reason}")


if __name__ == "__main__":
    main()