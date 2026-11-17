"""Catálogo fijo de estrategias usadas en las simulaciones.

Todas cumplen las reglas del torneo original de Axelrod (no inspeccionan ni
manipulan el código del rival ni el estado del juego) y son de ejecución
rápida (``long_run_time = False``). La lista es fija y ordenada para que las
simulaciones sean reproducibles.
"""

from __future__ import annotations

import axelrod as axl

# Las 11 clásicas pedidas explícitamente.
CLASSICS: list[type[axl.Player]] = [
    axl.TitForTat,
    axl.Cooperator,
    axl.Defector,
    axl.Grudger,
    axl.Random,
    axl.WinStayLoseShift,
    axl.TitFor2Tats,
    axl.SuspiciousTitForTat,
    axl.GTFT,
    axl.FirstByJoss,  # "Joss" del primer torneo de Axelrod (1980)
    axl.Alternator,
]

# 19 estrategias adicionales para llegar a 30, variadas en carácter:
# variantes de TFT, rencorosas, sondeadoras, explotadoras, Zero-Determinant
# y participantes del primer torneo de Axelrod.
EXTRAS: list[type[axl.Player]] = [
    axl.TwoTitsForTat,
    axl.HardTitForTat,
    axl.ContriteTitForTat,
    axl.AdaptiveTitForTat,
    axl.OmegaTFT,
    axl.Gradual,
    axl.SoftGrudger,
    axl.ForgetfulGrudger,
    axl.Prober,
    axl.Calculator,
    axl.Bully,
    axl.GoByMajority,
    axl.ZDExtort2,
    axl.ZDGTFT2,
    axl.FirstByTidemanAndChieruzzi,
    axl.FirstByNydegger,
    axl.FirstByDavis,
    axl.FirstByGrofman,
    axl.FirstByShubik,
]

FULL: list[type[axl.Player]] = CLASSICS + EXTRAS

SMALL: list[type[axl.Player]] = [
    axl.TitForTat,
    axl.Cooperator,
    axl.Defector,
    axl.Grudger,
    axl.Random,
]

PRESETS: dict[str, list[type[axl.Player]]] = {"small": SMALL, "full": FULL}


def get_strategies(preset: str) -> list[type[axl.Player]]:
    """Devuelve la lista de clases de estrategia de un preset."""
    try:
        return list(PRESETS[preset])
    except KeyError as exc:
        raise ValueError(f"Preset desconocido: {preset!r}. Opciones: {sorted(PRESETS)}") from exc


def _validate(strategies: list[type[axl.Player]]) -> None:
    names = [cls().name for cls in strategies]
    if len(set(names)) != len(names):
        raise ValueError("Hay estrategias duplicadas en la lista")
    for cls in strategies:
        player = cls()
        if not axl.Classifiers.obey_axelrod(player):
            raise ValueError(f"{player.name} no cumple las reglas de Axelrod")
        if axl.Classifiers["long_run_time"](player):
            raise ValueError(f"{player.name} es de ejecución lenta")


_validate(FULL)
assert len(FULL) == 30
