"""API de inferencia y demo interactiva.

Uso:
    uv run uvicorn tfm_axelrod.api:app --port 8000

Endpoints:
- ``GET /health``: estado y modelos cargados.
- ``POST /predict``: top-3 de estrategias más probables del humano.
- ``POST /play``: jugada de la IA en la ronda actual, pagos y marcadores.
- ``GET /``: demo web (tfm_axelrod/web/index.html).

La API no guarda estado: el cliente envía el historial completo en cada
llamada. La jugada de la IA se calcula SOLO con el historial previo
(``decide_ai_move`` ni siquiera recibe la jugada actual del humano).

Lógica de la IA:
- Rondas 1-5: agente Q-learning congelado (k=3, entrenado sin ruido).
- Desde la ronda 6: para cada una de las 30 estrategias calcula su
  puntuación esperada contra el humano, sum_h P(h) * head_to_head[s][h], y
  juega como la mejor. Para que esa estrategia decida de forma coherente con
  la partida real, se reconstruye desde cero y se le "reproduce" el historial
  real ronda a ronda (ver ``strategy_move``).
"""

from __future__ import annotations

import json
import os
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Literal

import axelrod as axl
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from tfm_axelrod.features import compute_features
from tfm_axelrod.strategies import FULL

ROOT = Path(os.environ.get("TFM_ROOT", Path(__file__).resolve().parent.parent))
CLASSIFIER_DIR = ROOT / "models" / "classifier_serving"
QTABLE_PATH = ROOT / "models" / "qlearning" / "q_k3_noise0.json"
H2H_PATH = ROOT / "models" / "head_to_head.json"
WEB_DIR = Path(__file__).resolve().parent / "web"

MIN_ROUNDS = 5
MAX_ROUNDS = 200
PAYOFF = {("C", "C"): (3, 3), ("C", "D"): (0, 5), ("D", "C"): (5, 0), ("D", "D"): (1, 1)}
ACTION = {"C": axl.Action.C, "D": axl.Action.D}
VERB = {"C": "coopero", "D": "deserto"}

Move = Literal["C", "D"]


# --------------------------------------------------------------------------- #
# Recursos (se cargan una vez)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def resources() -> dict:
    meta = json.loads((CLASSIFIER_DIR / "metadata.json").read_text(encoding="utf-8"))
    models = {n: joblib.load(CLASSIFIER_DIR / f"model_n{n}.joblib") for n in meta["n_values"]}
    qdata = json.loads(QTABLE_PATH.read_text(encoding="utf-8"))
    h2h = json.loads(H2H_PATH.read_text(encoding="utf-8"))
    classes = {cls().name: cls for cls in FULL}
    missing = set(meta["classes"]) ^ set(classes) | set(h2h["strategies"]) ^ set(classes)
    if missing:
        raise RuntimeError(f"Estrategias inconsistentes entre modelos y catálogo: {missing}")
    return {"meta": meta, "models": models, "q": qdata["q"], "memory": qdata["memory"],
            "h2h": h2h["scores"], "classes": classes}


# --------------------------------------------------------------------------- #
# Lógica
# --------------------------------------------------------------------------- #
def predict_strategy(human: list[str], ai: list[str]) -> dict:
    """Probabilidades del clasificador sobre la estrategia del humano.

    Usa el modelo del mayor n disponible <= rondas jugadas, con las
    características de las n primeras rondas (igual que en el entrenamiento).
    """
    res = resources()
    played = len(human)
    usable = [n for n in res["meta"]["n_values"] if n <= played]
    if not usable:
        raise ValueError(f"Hacen falta al menos {MIN_ROUNDS} rondas para predecir")
    n = max(usable)
    feats = compute_features(human[:n], ai[:n])  # perspectiva del humano
    X = pd.DataFrame([feats])[res["meta"]["features"]]
    model = res["models"][n]
    proba = dict(zip(model.classes_, model.predict_proba(X)[0].tolist()))
    top3 = sorted(proba.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return {"n": n, "rounds_used": n, "probabilities": proba,
            "top3": [{"strategy": s, "probability": round(p, 4)} for s, p in top3]}


def qlearning_move(ai: list[str], human: list[str]) -> str:
    """Acción voraz del agente Q-learning congelado (estado desde la IA).

    Equivale a ``QLearner(k, q, learning=False)``; se reimplementa aquí para no
    arrastrar dependencias de entrenamiento. Empates (no hay en esta tabla):
    coopera.
    """
    res = resources()
    k = res["memory"]
    state = "START" if not ai else ",".join(f"{a}{h}" for a, h in zip(ai[-k:], human[-k:]))
    q = res["q"].get(state, {"C": 0.0, "D": 0.0})
    return "D" if q["D"] > q["C"] else "C"


def strategy_move(cls: type[axl.Player], ai: list[str], human: list[str]) -> str:
    """Jugada de ``cls`` en la ronda actual, coherente con el historial real.

    Se crea la estrategia desde cero y se reproduce la partida: en cada ronda
    pasada se llama a ``strategy()`` (para que avance su estado interno, como
    en una partida real) pero se descarta su salida y se registra en ambos
    historiales lo que realmente se jugó. Aproximación: si la estrategia no
    controlaba la IA en alguna ronda pasada, su estado interno se basa en
    jugadas propias que no eligió ella (lo mismo que le ocurriría con ruido).
    Las estocásticas usan una semilla derivada del historial, así que la
    respuesta es reproducible. No se le informa de la longitud de la partida.
    """
    me, opp = cls(), axl.Cooperator()  # opp: solo sirve de contenedor del historial humano
    me.set_seed(zlib.crc32(("".join(ai) + "|" + "".join(human)).encode()))
    for a, h in zip(ai, human):
        me.strategy(opp)
        me.update_history(ACTION[a], ACTION[h])
        opp.update_history(ACTION[h], ACTION[a])
    return str(me.strategy(opp))


def decide_ai_move(ai: list[str], human: list[str]) -> dict:
    """Decide la jugada de la IA usando SOLO las rondas ya jugadas."""
    res = resources()
    rnd = len(ai) + 1
    if len(ai) < MIN_ROUNDS:
        move = qlearning_move(ai, human)
        why = (f"Ronda {rnd}: aún tengo pocos datos sobre ti, así que decide mi agente "
               f"Q-learning (memoria 3): {VERB[move]}.")
        return {"ai_move": move, "mode": "qlearning", "strategy_used": "Q-learning (k=3)",
                "expected_score": None, "prediction": None, "explanation": why}

    pred = predict_strategy(human, ai)
    expected = {s: sum(p * res["h2h"][s][h] for h, p in pred["probabilities"].items())
                for s in res["classes"]}
    best = max(sorted(expected), key=expected.get)  # desempate alfabético estable
    move = strategy_move(res["classes"][best], ai, human)
    guess = pred["top3"][0]
    why = (f"Creo que juegas como {guess['strategy']} ({guess['probability']:.0%}). "
           f"Contra eso, lo que más puntos espera es jugar como {best} "
           f"(≈{expected[best]:.2f} puntos por ronda). Siguiendo esa estrategia, {VERB[move]}.")
    return {"ai_move": move, "mode": "best_response", "strategy_used": best,
            "expected_score": round(expected[best], 4),
            "prediction": {"n": pred["n"], "top3": pred["top3"]}, "explanation": why}


# --------------------------------------------------------------------------- #
# Esquemas
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    human: list[Move] = Field(..., max_length=MAX_ROUNDS, description="Jugadas del humano")
    ai: list[Move] = Field(..., max_length=MAX_ROUNDS, description="Jugadas de la IA")

    @model_validator(mode="after")
    def same_length(self):
        if len(self.human) != len(self.ai):
            raise ValueError("human y ai deben tener la misma longitud")
        if len(self.human) < MIN_ROUNDS:
            raise ValueError(f"Hacen falta al menos {MIN_ROUNDS} rondas para predecir")
        return self


class StrategyProb(BaseModel):
    strategy: str
    probability: float


class Prediction(BaseModel):
    n: int
    top3: list[StrategyProb]


class PredictResponse(Prediction):
    rounds_played: int


class PlayRequest(BaseModel):
    human_history: list[Move] = Field(default_factory=list, max_length=MAX_ROUNDS - 1)
    ai_history: list[Move] = Field(default_factory=list, max_length=MAX_ROUNDS - 1)
    human_move: Move

    @model_validator(mode="after")
    def same_length(self):
        if len(self.human_history) != len(self.ai_history):
            raise ValueError("human_history y ai_history deben tener la misma longitud")
        return self


class Pair(BaseModel):
    human: int
    ai: int


class PlayResponse(BaseModel):
    round: int
    human_move: Move
    ai_move: Move
    payoffs: Pair
    scores: Pair
    prediction: Prediction | None
    mode: Literal["qlearning", "best_response"]
    strategy_used: str
    expected_score: float | None
    explanation: str


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="TFM Axelrod: API de inferencia", version="1.0.0")


@app.get("/health")
def health() -> dict:
    res = resources()
    return {"status": "ok", "classifier_n_values": res["meta"]["n_values"],
            "classifier_algorithm": res["meta"]["algorithm"],
            "strategies": len(res["classes"]), "qlearning_memory": res["memory"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> dict:
    pred = predict_strategy(req.human, req.ai)
    return {"n": pred["n"], "top3": pred["top3"], "rounds_played": len(req.human)}


@app.post("/play", response_model=PlayResponse)
def play(req: PlayRequest) -> dict:
    human, ai = list(req.human_history), list(req.ai_history)
    decision = decide_ai_move(ai, human)  # no recibe req.human_move
    ph, pa = PAYOFF[(req.human_move, decision["ai_move"])]
    sh = sum(PAYOFF[(h, a)][0] for h, a in zip(human, ai)) + ph
    sa = sum(PAYOFF[(h, a)][1] for h, a in zip(human, ai)) + pa
    return {"round": len(human) + 1, "human_move": req.human_move,
            "payoffs": {"human": ph, "ai": pa}, "scores": {"human": sh, "ai": sa},
            **decision}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    path = WEB_DIR / "index.html"
    if not path.exists():
        raise HTTPException(404, "Falta la demo web")
    return FileResponse(path, media_type="text/html; charset=utf-8")
