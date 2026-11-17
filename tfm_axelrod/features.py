"""Características de comportamiento de un jugador en las primeras n rondas.

Réplica en Python puro de la celda 2 de ``pipelines/02_curated_to_analytics.py``
(Spark). Sirve para calcular, fuera de Databricks, las mismas características
con las que se entrenó el clasificador (p. ej. desde la API).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Valores de n (rondas observadas) para los que existe un modelo.
N_VALUES: list[int] = [5, 10, 20, 30, 50, 75, 100]

# Estados conjuntos (jugada propia, jugada del rival), 1 = coopera.
STATES: dict[str, tuple[int, int]] = {"cc": (1, 1), "cd": (1, 0), "dc": (0, 1), "dd": (0, 0)}

# Columnas del dataset que NO son características.
META_COLUMNS: list[str] = [
    "match_key", "player_index", "player", "opponent", "noise", "turns",
    "repetition", "is_self_play", "n",
]

# Orden canónico de las características (el mismo que en el parquet).
FEATURE_COLUMNS: list[str] = [
    "coop_rate", "opp_coop_rate", "first_move_coop", "avg_payoff", "switch_rate",
    "max_c_streak", "max_d_streak", "first_defection", "state_entropy",
    *[f"p_coop_after_{s}" for s in STATES],
    *[f"freq_after_{s}" for s in STATES],
]

# Pago del jugador según (c, oc): R=3, S=0, T=5, P=1.
PAYOFF: dict[tuple[int, int], int] = {(1, 1): 3, (1, 0): 0, (0, 1): 5, (0, 0): 1}


def _to_bits(actions: Sequence[str], who: str) -> list[int]:
    bits = []
    for a in actions:
        a = str(a).upper()
        if a not in ("C", "D"):
            raise ValueError(f"Acción no válida en {who}: {a!r} (se espera 'C' o 'D')")
        bits.append(1 if a == "C" else 0)
    return bits


def _max_streak(bits: list[int], value: int) -> int:
    best = cur = 0
    for b in bits:
        cur = cur + 1 if b == value else 0
        best = max(best, cur)
    return best


def compute_features(own_actions: Sequence[str], opp_actions: Sequence[str]) -> dict[str, float]:
    """Calcula las características a partir de dos secuencias de "C"/"D".

    ``own_actions`` son las jugadas del jugador a clasificar y ``opp_actions``
    las de su rival, ronda a ronda, con la misma longitud ``n >= 1``. Devuelve
    un dict con las claves de :data:`FEATURE_COLUMNS` (en ese orden).

    Con ``n == 1`` las tasas que dividen entre ``n - 1`` valen 0.
    """
    c = _to_bits(own_actions, "own_actions")
    oc = _to_bits(opp_actions, "opp_actions")
    n = len(c)
    if n == 0:
        raise ValueError("Las secuencias no pueden estar vacías")
    if len(oc) != n:
        raise ValueError(f"Longitudes distintas: {n} jugadas propias y {len(oc)} del rival")
    trans = n - 1  # nº de rondas con ronda anterior

    switches = sum(c[r] != c[r - 1] for r in range(1, n))
    first_d = next((r + 1 for r, b in enumerate(c) if b == 0), n + 1)

    now_counts = dict.fromkeys(STATES, 0)
    after_counts = dict.fromkeys(STATES, 0)
    coop_after = dict.fromkeys(STATES, 0)
    by_state = {v: k for k, v in STATES.items()}
    for r in range(n):
        now_counts[by_state[(c[r], oc[r])]] += 1
        if r > 0:
            prev = by_state[(c[r - 1], oc[r - 1])]
            after_counts[prev] += 1
            coop_after[prev] += c[r]

    entropy = 0.0
    for k in STATES:
        p = now_counts[k] / n
        if p > 0:
            entropy -= p * math.log2(p)

    feats: dict[str, float] = {
        "coop_rate": sum(c) / n,
        "opp_coop_rate": sum(oc) / n,
        "first_move_coop": c[0],
        "avg_payoff": sum(PAYOFF[(a, b)] for a, b in zip(c, oc)) / n,
        "switch_rate": switches / trans if trans else 0.0,
        "max_c_streak": _max_streak(c, 1) / n,
        "max_d_streak": _max_streak(c, 0) / n,
        "first_defection": first_d / n,
        "state_entropy": entropy,
    }
    for k in STATES:
        feats[f"p_coop_after_{k}"] = coop_after[k] / after_counts[k] if after_counts[k] else 0.5
    for k in STATES:
        feats[f"freq_after_{k}"] = after_counts[k] / trans if trans else 0.0
    return feats
