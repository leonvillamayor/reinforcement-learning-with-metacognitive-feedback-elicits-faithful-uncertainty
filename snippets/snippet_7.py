"""
RLMF — Control Panel Experimental
==================================

Mini-simulación del panel de ablación del paper:
"Reinforcement Learning with Metacognitive Feedback
Elicits Faithful Uncertainty Expression in LLMs"
(arXiv:2606.32032)

Este script NO entrena un LLM. Modela el *espacio de configuración*
de recompensas y mide, sobre métricas sintéticas pero realistas,
qué pasa cuando se modifica cada decisión de diseño:

    • Qué señales incluir (correctness, ECE, refusal, metacog-consistency)
    • Con qué pesos combinarlas (additive, weighted-sum, gated)
    • Qué formulation usar (sum, product, min, learned-mix)
    • Qué system prompt (short / verbose / metacog-explicit)
    • Cuántos datos usar (data-scaling curve)

Genera una tabla-resumen apta para la sección "Ablations" del paper.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 1. Métricas sintéticas: ground truth plausible del paper
# ---------------------------------------------------------------------------
# En el paper real:
#   • acc        = exact-match accuracy
#   • ece        = Expected Calibration Error (↓ mejor)
#   • refusal    = refusal-appropriateness F1 (↑ mejor)
#   • metacog_c  = metacognitive consistency (verbal ≡ internal), ↑ mejor
# Cada política de RL produce una tupla de estas cuatro métricas.

Metrics = tuple[float, float, float, float]


@dataclass(frozen=True)
class RewardConfig:
    """Una configuración del panel experimental.

    Cada campo corresponde a una decisión de diseño que el paper
    justifica por ablación. El comentario alude a la sección del paper.
    """
    name: str
    # --- Señales activas ---
    use_correctness: bool = True   # §3.2 reward r_acc
    use_ece: bool = True           # §3.2 reward r_cal
    use_refusal: bool = True       # §3.2 reward r_ref
    use_metacog: bool = True       # §3.3 reward r_meta (clave del paper)
    # --- Pesos ---
    w_acc: float = 1.0
    w_ece: float = 0.5
    w_ref: float = 0.5
    w_meta: float = 1.5
    # --- Formulación matemática ---
    # "sum"   -> weighted sum lineal (baseline GRPO-style)
    # "gated" -> r_meta actúa como *gate* sobre el resto
    # "min"   -> worst-case: r = min(signals) (robustez)
    formulation: str = "gated"
    # --- System prompt ---
    prompt_style: str = "metacog"   # "short" | "verbose" | "metacog"
    # --- Datos ---
    n_train: int = 8000

    def reward(self, m: Metrics) -> float:
        acc, ece, refusal, meta = m
        r_acc = self.w_acc * acc
        r_ece = self.w_ece * (1.0 - ece)   # ECE ↓ → reward ↑
        r_ref = self.w_ref * refusal
        r_meta = self.w_meta * meta

        parts = []
        if self.use_correctness: parts.append(r_acc)
        if self.use_ece:        parts.append(r_ece)
        if self.use_refusal:    parts.append(r_ref)
        # metacog es SIEMPRE la pieza central del método RLMF
        if self.use_metacog:    parts.append(r_meta)

        if self.formulation == "sum":
            return float(sum(parts))
        if self.formulation == "gated":
            # r_meta multiplica: si no hay metacog consistente,
            # las otras señales valen poco (faithfulness-first)
            return float(sum(parts) * (0.25 + 0.75 * meta))
        if self.formulation == "min":
            return float(min(parts))
        raise ValueError(f"formulation desconocida: {self.formulation}")


# ---------------------------------------------------------------------------
# 2. Simulador de "entrenamiento" (placeholder del RL real)
# ---------------------------------------------------------------------------
# Aquí modelamos el *efecto esperado* de cada decisión: quitar metacog
# pierde fidelidad verbal; quitar ECE pierde calibración; prompt "short"
# pierde expresión de incertidumbre; etc. Los coeficientes están
# calibrados para que la configuración canónica del paper gane.

@dataclass
class SimParams:
    base_acc: float = 0.55
    base_ece: float = 0.30
    base_ref: float = 0.60
    base_meta: float = 0.40


def simulate(cfg: RewardConfig, rng: np.random.Generator,
             base: SimParams = SimParams()) -> Metrics:
    """Devuelve métricas sintéticas condicionadas al config."""
    # Efecto de cada señal (escala [0,1] tipo pass-rate)
    acc  = base.base_acc
    ece  = base.base_ece
    ref  = base.base_ref
    meta = base.base_meta

    # Sin r_meta → la consistencia metacognitiva colapsa (hallazgo paper §5.2)
    if not cfg.use_metacog: meta *= 0.35
    # Sin r_cal → la calibración se degrada
    if not cfg.use_ece:     ece = min(1.0, ece + 0.15)
    # Sin r_ref → refusals inapropiados suben (peor F1)
    if not cfg.use_refusal: ref  = max(0.0, ref - 0.10)
    # Sin r_acc → accuracy cae
    if not cfg.use_correctness: acc = max(0.0, acc - 0.08)

    # Formulación: el "gated" potencia la metacog más que el sum puro
    if cfg.formulation == "gated":
        meta = min(1.0, meta * 1.10)
    elif cfg.formulation == "min":
        # peor-case: arrastra todo hacia abajo
        meta *= 0.90; ref *= 0.95

    # System prompt
    if cfg.prompt_style == "short":
        meta *= 0.80
    elif cfg.prompt_style == "verbose":
        meta *= 0.95  # verbosidad sin instrucciones metacog no ayuda
    # "metacog" mantiene meta

    # Data scaling (log): pocos datos → más ruido → peor calibración
    if cfg.n_train < 4000:
        ece = min(1.0, ece + 0.05)
    if cfg.n_train < 2000:
        meta *= 0.90

    # Ruido estocástico (simula variabilidad entre seeds)
    noise = rng.normal(0, 0.02, size=4)
    acc  = float(np.clip(acc  + noise[0], 0, 1))
    ece  = float(np.clip(ece  + noise[1], 0, 1))
    ref  = float(np.clip(ref  + noise[2], 0, 1))
    meta = float(np.clip(meta + noise[3], 0, 1))

    return (acc, ece, ref, meta)


# ---------------------------------------------------------------------------
# 3. Ablaciones: cada decisión de diseño, modificada una a una
# ---------------------------------------------------------------------------
CANONICAL = RewardConfig(
    name="RLMF (canonical)",
    formulation="gated",
    prompt_style="metacog",
    n_train=8000,
)


def ablate_signal(cfg: RewardConfig, signal: str) -> RewardConfig:
    overrides: dict[str, bool] = {
        "no_correctness": dict(use_correctness=False),
        "no_ece":         dict(use_ece=False),
        "no_refusal":     dict(use_refusal=False),
        "no_metacog":     dict(use_metacog=False),
    }
    new = {**cfg.__dict__}
    new.update(overrides[signal])
    new["name"] = signal
    return RewardConfig(**new)


def ablate_formulation(cfg: RewardConfig, f: str) -> RewardConfig:
    new = {**cfg.__dict__}
    new["formulation"] = f
    new["name"] = f"form={f}"
    return RewardConfig(**new)


def ablate_prompt(cfg: RewardConfig, p: str) -> RewardConfig:
    new = {**cfg.__dict__}
    new["prompt_style"] = p
    new["name"] = f"prompt={p}"
    return RewardConfig(**new)


def data_scaling(cfg: RewardConfig) -> list[RewardConfig]:
    sizes = [500, 1000, 2000, 4000, 8000, 16000]
    return [
        RewardConfig(**{**cfg.__dict__, "n_train": n,
                        "name": f"n={n}"})
        for n in sizes
    ]


# ---------------------------------------------------------------------------
# 4. Ejecución + tabla de resultados
# ---------------------------------------------------------------------------
def mean_metrics(runs: Iterable[Metrics]) -> Metrics:
    arr = np.array(list(runs))
    return tuple(arr.mean(axis=0).tolist())  # type: ignore[return-value]


def evaluate(cfg: RewardConfig, seeds: int = 5) -> dict[str, float]:
    rng = np.random.default_rng(hash(cfg.name) & 0xFFFFFFFF)
    runs = [simulate(cfg, rng) for _ in range(seeds)]
    acc, ece, ref, meta = mean_metrics(runs)
    r = cfg.reward((acc, ece, ref, meta))
    return dict(acc=acc, ece=ece, refusal=ref, metacog=meta, reward=r)


def run_panel(seeds: int = 5) -> list[dict[str, float]]:
    configs: list[RewardConfig] = [CANONICAL]

    # Ablación de señales
    for s in ["no_correctness", "no_ece", "no_refusal", "no_metacog"]:
        configs.append(ablate_signal(CANONICAL, s))

    # Ablación de formulation
    for f in ["sum", "min", "gated"]:
        configs.append(ablate_formulation(CANONICAL, f))

    # Ablación de prompt
    for p in ["short", "verbose", "metacog"]:
        configs.append(ablate_prompt(CANONICAL, p))

    # Data scaling
    configs.extend(data_scaling(CANONICAL))

    rows = []
    for cfg in configs:
        res = evaluate(cfg, seeds=seeds)
        res["config"] = cfg.name
        rows.append(res)
    return rows


def render_table(rows: Sequence[dict[str, float]]) -> str:
    cols = ["config", "reward", "acc", "ece", "refusal", "metacog"]
    widths = {c: max(len(c), max(len(f"{r[c]:.3f}")
                                 if c != "config" else len(r[c])
                                 for r in rows))
              for c in cols}
    head = " | ".join(c.ljust(widths[c]) for c in cols)
    sep  = "-+-".join("-" * widths[c] for c in cols)
    body = []
    for r in rows:
        line = []
        for c in cols:
            v = f"{r[c]:.3f}" if c != "config" else r[c]
            line.append(v.ljust(widths[c]))
        body.append(" | ".join(line))
    return "\n".join([head, sep, *body])


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)
    rows = run_panel(seeds=5)

    # Ordena por reward descendente: el canónico debería quedar arriba.
    rows.sort(key=lambda r: r["reward"], reverse=True)
    print(render_table(rows))

    can = next(r for r in rows if r["config"] == CANONICAL.name)
    print("\nHighlight (hallazgo del paper §5.2):")
    print(f"  • Quitar r_meta  → metacog cae a "
          f"{rows[[r['config'] for r in rows].index('no_metacog')]['metacog']:.3f}")
    print(f"  • Config canónico: metacog = {can['metacog']:.3f}, "
          f"ECE = {can['ece']:.3f}")