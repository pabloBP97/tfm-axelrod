import random
import warnings

import axelrod as axl
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # aviso de Starlette sobre httpx, ajeno al proyecto
    from fastapi.testclient import TestClient

from tfm_axelrod.api import app, qlearning_move, resources, strategy_move
from tfm_axelrod.qlearning import QLearner, state_key

client = TestClient(app)
C, D = axl.Action.C, axl.Action.D


def play_game(rounds: int, human_policy) -> list[dict]:
    human, ai, out = [], [], []
    for _ in range(rounds):
        move = human_policy(human, ai)
        r = client.post("/play", json={"human_history": human, "ai_history": ai,
                                       "human_move": move})
        assert r.status_code == 200, r.text
        body = r.json()
        human.append(body["human_move"])
        ai.append(body["ai_move"])
        out.append(body)
    return out


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["strategies"] == 30


def test_index_page():
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Cooperar" in r.text and "Lo que la IA cree que eres" in r.text


def test_full_game_and_prediction_from_round_6():
    rng = random.Random(0)
    res = play_game(30, lambda h, a: rng.choice("CD"))
    sh = sa = 0
    for i, r in enumerate(res, start=1):
        assert r["round"] == i
        assert r["ai_move"] in ("C", "D")
        payoff = {("C", "C"): (3, 3), ("C", "D"): (0, 5), ("D", "C"): (5, 0), ("D", "D"): (1, 1)}
        ph, pa = payoff[(r["human_move"], r["ai_move"])]
        assert (r["payoffs"]["human"], r["payoffs"]["ai"]) == (ph, pa)
        sh, sa = sh + ph, sa + pa
        assert (r["scores"]["human"], r["scores"]["ai"]) == (sh, sa)
        assert r["explanation"]
        if i <= 5:
            assert r["prediction"] is None and r["mode"] == "qlearning"
        else:
            assert r["mode"] == "best_response"
            top = r["prediction"]["top3"]
            assert len(top) == 3 and r["prediction"]["n"] <= i - 1
            assert all(0 <= t["probability"] <= 1 for t in top)
            assert top[0]["probability"] >= top[1]["probability"] >= top[2]["probability"]
            assert r["strategy_used"] in resources()["classes"]


@pytest.mark.parametrize("seed", range(6))
def test_ai_move_ignores_current_human_move(seed):
    rng = random.Random(seed)
    n = rng.randint(0, 40)
    human = [rng.choice("CD") for _ in range(n)]
    ai = [rng.choice("CD") for _ in range(n)]
    a = client.post("/play", json={"human_history": human, "ai_history": ai, "human_move": "C"}).json()
    b = client.post("/play", json={"human_history": human, "ai_history": ai, "human_move": "D"}).json()
    for key in ("ai_move", "mode", "strategy_used", "prediction", "explanation"):
        assert a[key] == b[key]


def test_predict_endpoint_and_n_selection():
    human, ai = list("CCDCDCCDCDCC"), list("CCCDCDCCDCDC")
    r = client.post("/predict", json={"human": human, "ai": ai})
    assert r.status_code == 200
    assert r.json()["n"] == 10 and len(r.json()["top3"]) == 3
    assert client.post("/predict", json={"human": human[:5], "ai": ai[:5]}).json()["n"] == 5


@pytest.mark.parametrize("url,payload", [
    ("/predict", {"human": list("CCCC"), "ai": list("CCCC")}),                # < 5 rondas
    ("/predict", {"human": list("CCCCC"), "ai": list("CCCC")}),               # longitudes distintas
    ("/predict", {"human": list("CCCCX"), "ai": list("CCCCC")}),              # acción no válida
    ("/play", {"human_history": ["C"], "ai_history": [], "human_move": "C"}),
    ("/play", {"human_history": [], "ai_history": [], "human_move": "X"}),
    ("/play", {"human_history": [], "ai_history": []}),                       # falta human_move
    ("/play", {"human_history": ["c"], "ai_history": ["C"], "human_move": "C"}),
    ("/play", {"human_history": ["C"] * 200, "ai_history": ["C"] * 200, "human_move": "C"}),
])
def test_invalid_inputs(url, payload):
    assert client.post(url, json=payload).status_code == 422


def test_qlearning_move_matches_frozen_agent():
    res = resources()
    q = {s: [v["C"], v["D"]] for s, v in res["q"].items()}
    agent = QLearner(res["memory"], q, learning=False)
    rng = random.Random(1)
    for _ in range(50):
        n = rng.randint(0, 6)
        ai = [rng.choice("CD") for _ in range(n)]
        human = [rng.choice("CD") for _ in range(n)]
        s = state_key(ai, human, res["memory"])
        assert qlearning_move(ai, human) == agent.greedy_action(s)


def test_strategy_move_uses_real_history():
    # Tit For Tat copia la última jugada real del humano.
    assert strategy_move(axl.TitForTat, list("CCD"), list("CDC")) == "C"
    assert strategy_move(axl.TitForTat, list("CCC"), list("CCD")) == "D"
    # Grudger guarda rencor aunque la IA jugara C después (historial forzado).
    assert strategy_move(axl.Grudger, list("CCCC"), list("CDCC")) == "D"
    # Alternator alterna según su propio historial real.
    assert strategy_move(axl.Alternator, list("CD"), list("CC")) == "C"
    # Estocásticas: misma entrada -> misma salida.
    h, a = list("CDCCDDCD"), list("CCDCDCDD")
    assert len({strategy_move(axl.Random, a, h) for _ in range(5)}) == 1
