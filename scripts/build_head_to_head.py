"""Tabla de enfrentamientos para la API.

Calcula, a partir de data/raw/roundrobin (solo ruido 0 y 200 rondas), la
puntuación media por ronda de cada estrategia contra cada rival (media sobre
repeticiones y rondas) y la guarda en models/head_to_head.json.

Uso:
    uv run python scripts/build_head_to_head.py
"""

import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds

RAW = Path("data/raw/roundrobin")
OUT = Path("models/head_to_head.json")

table = ds.dataset(RAW, format="parquet").to_table(
    columns=["player", "opponent", "player_payoff"],
    filter=(pc.field("noise") == 0.0) & (pc.field("turns") == 200),
)
df = table.to_pandas()
if df.empty:
    raise SystemExit("No hay datos con ruido 0 y 200 rondas en data/raw/roundrobin")

scores = df.groupby(["player", "opponent"]).player_payoff.mean().unstack()
strategies = sorted(scores.index)
if sorted(scores.columns) != strategies or scores.isna().any().any():
    raise SystemExit("La tabla no está completa: faltan enfrentamientos")

payload = {
    "description": "Puntuación media por ronda de 'strategy' (fila) contra 'opponent' (columna)",
    "noise": 0.0,
    "turns": 200,
    "strategies": strategies,
    "scores": {s: {o: round(float(scores.at[s, o]), 4) for o in strategies} for s in strategies},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"{OUT}: {len(strategies)} estrategias, {len(df)} jugadas")
