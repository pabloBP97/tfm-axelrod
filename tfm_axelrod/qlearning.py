"""Agente de Q-learning tabular para el dilema del prisionero iterado.

Uso:
    uv run python -m tfm_axelrod.qlearning

Pregunta: ¿un agente que aprende desde cero, sin ninguna estructura previa,
redescubre comportamientos tipo Tit-for-Tat (no desertar primero, represalia,
perdón)?

- Estado: las últimas ``k`` rondas (jugada propia y del rival). Mientras aún
  no hay ``k`` rondas, el estado es la historia parcial ("START" en la ronda
  1), que actúa como estado especial de apertura.
- Acciones: C o D. Recompensa: pago de la ronda (R=3, S=0, T=5, P=1).
- Política epsilon-greedy con decaimiento exponencial por episodio.
- ``QLearner`` es un ``axl.Player``: en modo aprendizaje actualiza su tabla
  durante la partida; en modo congelado juega la política voraz y no la toca.

Para cada k en {1, 2, 3} y ruido en {0, 0.05} entrena con episodios de 200
rondas contra rivales del preset ``full`` elegidos al azar, evalúa el agente
congelado en un torneo de 31 jugadores, mide su comportamiento, lo pasa por el
clasificador y lo registra todo en MLflow.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import axelrod as axl
import numpy as np
import pandas as pd

from tfm_axelrod.simulate import match_seed
from tfm_axelrod.strategies import FULL

C, D = axl.Action.C, axl.Action.D
ACTIONS = (C, D)  # índice 0 = C, 1 = D
PAYOFF = {(C, C): 3, (C, D): 0, (D, C): 5, (D, D): 1}
START = "START"

MEMORIES = [1, 2, 3]
NOISES = [0.0, 0.05]
TURNS = 200
EVAL_REPETITIONS = 10
HYPERPARAMS = {"alpha": 0.1, "gamma": 0.95, "epsilon_start": 1.0, "epsilon_end": 0.01,
               "epsilon_decay_fraction": 0.8}
REFERENCES = ["Tit For Tat", "Win-Stay Lose-Shift", "Grudger", "Defector"]


# --------------------------------------------------------------------------- #
# Agente
# --------------------------------------------------------------------------- #
def state_key(own: list, opp: list, k: int) -> str:
    """Estado = últimas k rondas, p. ej. "CD,DD" (propia+rival por ronda)."""
    if not own:
        return START
    pairs = zip(own[-k:], opp[-k:])
    return ",".join(f"{a}{b}" for a, b in pairs)


class QLearner(axl.Player):
    """Jugador de Q-learning tabular.

    La tabla ``q_table`` (dict estado -> [Q(C), Q(D)]) se pasa como argumento
    para que sobreviva a ``reset()``, que axelrod llama al inicio de cada
    partida reconstruyendo el jugador con sus ``init_kwargs``.
    """

    name = "Q-learning"
    classifier = {
        "memory_depth": 1, "stochastic": True, "long_run_time": False,
        "inspects_source": False, "manipulates_source": False, "manipulates_state": False,
    }

    def __init__(self, memory: int = 1, q_table: dict | None = None, learning: bool = True,
                 alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1,
                 visits: dict | None = None, label: str | None = None) -> None:
        super().__init__()
        self.memory = memory
        self.q = q_table if q_table is not None else {}
        self.visits = visits if visits is not None else {}
        self.learning = learning
        self.alpha, self.gamma = alpha, gamma
        self.epsilon = epsilon if learning else 0.0
        self.name = label or f"Q-learning (k={memory})"
        self.classifier["memory_depth"] = float("inf") if learning else memory
        # Congelado es determinista salvo empates (estados sin preferencia).
        self.classifier["stochastic"] = learning or self._has_ties()
        self._pending: tuple[str, int] | None = None  # (estado, acción) sin actualizar

    def __repr__(self) -> str:  # evita volcar la tabla Q en el nombre
        return self.name

    def _has_ties(self) -> bool:
        return any(v[0] == v[1] for v in self.q.values())

    def _values(self, s: str) -> list[float]:
        if self.learning:
            return self.q.setdefault(s, [0.0, 0.0])
        return self.q.get(s, [0.0, 0.0])

    def update(self, s: str, a: int, r: float, s_next: str | None) -> None:
        """Q(s,a) += alpha * (r + gamma * max Q(s',·) - Q(s,a)); s_next=None es terminal."""
        q = self._values(s)
        target = r if s_next is None else r + self.gamma * max(self._values(s_next))
        q[a] += self.alpha * (target - q[a])

    def _learn_from_last_round(self, s_next: str | None) -> None:
        if self.learning and self._pending is not None:
            s, a = self._pending
            r = PAYOFF[(self.history[-1], self.history.coplays[-1])]
            self.update(s, a, r, s_next)

    def strategy(self, opponent: axl.Player) -> axl.Action:
        s = state_key(self.history, self.history.coplays, self.memory)
        self._learn_from_last_round(s)
        if self.learning and self._random.random() < self.epsilon:
            a = 0 if self._random.random() < 0.5 else 1
        else:
            qc, qd = self._values(s)
            a = (0 if self._random.random() < 0.5 else 1) if qc == qd else int(qd > qc)
        if self.learning:
            self.visits[s] = self.visits.get(s, 0) + 1
            self._pending = (s, a)
        return ACTIONS[a]

    def finish_episode(self) -> None:
        """Actualización terminal con el pago de la última ronda."""
        self._learn_from_last_round(None)
        self._pending = None

    def greedy_action(self, s: str) -> str | None:
        """Acción voraz en el estado s (None si no hay preferencia)."""
        qc, qd = self.q.get(s, [0.0, 0.0])
        return None if qc == qd else ("C" if qc > qd else "D")


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def config_seed(seed: int, memory: int, noise: float) -> int:
    return zlib.crc32(f"{seed}|qlearning|{memory}|{noise!r}".encode())


def train(memory: int, noise: float, episodes: int, seed: int,
          hp: dict = HYPERPARAMS, opponents: list = FULL) -> dict:
    """Entrena un agente; devuelve tabla Q, visitas y curva de recompensa por episodio."""
    rng = np.random.default_rng(config_seed(seed, memory, noise))
    q, visits = {}, {}
    decay = (hp["epsilon_end"] / hp["epsilon_start"]) ** (
        1 / max(1.0, hp["epsilon_decay_fraction"] * episodes))
    curve = np.empty(episodes, dtype=np.float64)
    for ep in range(episodes):
        eps = max(hp["epsilon_end"], hp["epsilon_start"] * decay ** ep)
        agent = QLearner(memory, q, True, hp["alpha"], hp["gamma"], eps, visits)
        opp = opponents[rng.integers(len(opponents))]()
        match = axl.Match((agent, opp), turns=TURNS, noise=noise,
                          seed=int(rng.integers(2**31)))
        match.play()
        agent.finish_episode()
        curve[ep] = np.mean([s[0] for s in match.scores()])
    return {"q": q, "visits": visits, "curve": curve}


# --------------------------------------------------------------------------- #
# Evaluación
# --------------------------------------------------------------------------- #
def play(p1: axl.Player, p2: axl.Player, noise: float, rep: int, seed: int):
    m = axl.Match((p1, p2), turns=TURNS, noise=noise,
                  seed=match_seed(seed, noise, TURNS, rep, p1.name, p2.name))
    inter = m.play()
    sc = np.array(m.scores(), dtype=float)
    own = "".join(str(a) for a, _ in inter)
    opp = "".join(str(b) for _, b in inter)
    return sc[:, 0].mean(), sc[:, 1].mean(), own, opp


def behaviour(matches: list[tuple[str, str]]) -> dict[str, float]:
    """Métricas de comportamiento a partir de pares (jugadas propias, del rival)."""
    coop, first, dd_n, dd_d, cc_n, cc_c = [], [], 0, 0, 0, 0
    for own, opp in matches:
        coop.append(own.count("C") / len(own))
        f_own, f_opp = own.find("D"), opp.find("D")
        f_own = len(own) if f_own < 0 else f_own
        f_opp = len(opp) if f_opp < 0 else f_opp
        first.append(f_own < f_opp)
        for t in range(1, len(own)):
            if opp[t - 1] == "D":
                dd_n += 1
                dd_d += own[t] == "D"
            else:
                cc_n += 1
                cc_c += own[t] == "C"
    return {
        "coop_rate": float(np.mean(coop)),
        "defects_first_pct": float(np.mean(first)),
        "p_defect_after_opp_defect": dd_d / dd_n if dd_n else float("nan"),
        "p_coop_after_opp_coop": cc_c / cc_n if cc_n else float("nan"),
    }


def baseline(noise: float, seed: int, strategies: list = FULL) -> dict:
    """Todos contra todos entre las 30 estrategias (sin autopartidas)."""
    names = [cls().name for cls in strategies]
    n = len(names)
    S = np.zeros((n, n))
    ref = {r: [] for r in REFERENCES}
    for i in range(n):
        for j in range(i + 1, n):
            for rep in range(EVAL_REPETITIONS):
                a, b, own, opp = play(strategies[i](), strategies[j](), noise, rep, seed)
                S[i, j] += a / EVAL_REPETITIONS
                S[j, i] += b / EVAL_REPETITIONS
                if names[i] in ref:
                    ref[names[i]].append((own, opp))
                if names[j] in ref:
                    ref[names[j]].append((opp, own))
    return {"noise": noise, "names": names, "S": S,
            "reference_behaviour": {r: behaviour(m) for r, m in ref.items()}}


def evaluate_agent(agent: QLearner, noise: float, seed: int, strategies: list = FULL) -> dict:
    """Agente congelado contra cada estrategia (sus filas/columnas en el torneo)."""
    row, col, matches = [], [], []
    for cls in strategies:
        a_sum = b_sum = 0.0
        for rep in range(EVAL_REPETITIONS):
            a, b, own, opp = play(agent.clone(), cls(), noise, rep, seed)
            a_sum += a
            b_sum += b
            matches.append((own, opp))
        row.append(a_sum / EVAL_REPETITIONS)
        col.append(b_sum / EVAL_REPETITIONS)
    return {"row": np.array(row), "col": np.array(col), "matches": matches}


def ranking(base: dict, agent_name: str, ev: dict) -> pd.DataFrame:
    """Puntuación media por ronda (sin autopartida) y posición entre los 31."""
    n = len(base["names"])
    S = np.zeros((n + 1, n + 1))
    S[:n, :n] = base["S"]
    S[n, :n] = ev["row"]
    S[:n, n] = ev["col"]
    score = S.sum(axis=1) / n  # n rivales por jugador
    df = pd.DataFrame({"strategy": base["names"] + [agent_name], "score": score})
    df["rank"] = df.score.rank(ascending=False, method="min").astype(int)
    return df.sort_values("rank").reset_index(drop=True)


def classify(matches: list[tuple[str, str]], model_dir: Path, n: int = 100) -> Counter:
    import joblib

    from tfm_axelrod.features import compute_features

    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(model_dir / f"model_n{n}.joblib")
    X = pd.DataFrame([compute_features(list(o[:n]), list(p[:n])) for o, p in matches])
    return Counter(model.predict(X[meta["features"]]))


def run_config(memory: int, noise: float, episodes: int, seed: int) -> dict:
    """Entrena y evalúa una configuración (se ejecuta en un proceso aparte)."""
    t0 = time.perf_counter()
    tr = train(memory, noise, episodes, seed)
    train_s = time.perf_counter() - t0
    agent = QLearner(memory, tr["q"], learning=False)
    t1 = time.perf_counter()
    ev = evaluate_agent(agent, noise, seed)
    return {"memory": memory, "noise": noise, **tr, "eval": ev, "name": agent.name,
            "train_seconds": train_s, "eval_seconds": time.perf_counter() - t1}


def _baseline_task(noise: float, seed: int) -> dict:
    return baseline(noise, seed)


# --------------------------------------------------------------------------- #
# Salidas
# --------------------------------------------------------------------------- #
def k1_policy_table(results: list[dict]) -> pd.DataFrame:
    states = [START, "CC", "CD", "DC", "DD"]
    rows = []
    for r in results:
        if r["memory"] != 1:
            continue
        ag = QLearner(1, r["q"], learning=False)
        rows.append({"policy": f"Q-learning (ruido {r['noise']:g})",
                     **{s: ag.greedy_action(s) or "=" for s in states},
                     **{f"Q_{s}": "{:.2f}/{:.2f}".format(*r["q"].get(s, [0, 0])) for s in states}})
    ref = {  # (inicio, CC, CD, DC, DD); estado = (jugada propia, jugada del rival)
        "Tit For Tat": "CCDCD", "Win-Stay Lose-Shift": "CCDDC",
        "Grudger": "CCDDD", "Defector": "DDDDD",
    }
    for name, acts in ref.items():
        rows.append({"policy": name, **dict(zip(states, acts))})
    return pd.DataFrame(rows)


def make_figures(results, curves_df, best, rank_df, policy, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from tfm_axelrod.plot_style import AXIS, INK, INK2, SERIES, SURFACE, apply_style

    apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(fig_dir / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Figura 10: curvas de aprendizaje
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ends = []
    for i, (cfg, g) in enumerate(curves_df.groupby("config", sort=False)):
        ax.plot(g.episode, g.reward_ma, color=SERIES[i], lw=1.8)
        ends.append((g.reward_ma.iloc[-1], cfg))
    ax.set_xlim(0, curves_df.episode.max() * 1.2)
    lo, hi = ax.get_ylim()
    prev = None
    for y, label in sorted(ends, reverse=True):
        y = y if prev is None else min(y, prev - (hi - lo) * 0.045)
        ax.text(curves_df.episode.max() * 1.01, y, label, va="center", fontsize=9, color=INK2)
        prev = y
    ax.set_xlabel("Episodio (partida de 200 rondas)")
    ax.set_ylabel("Recompensa media por ronda (media móvil)")
    ax.set_title("Curvas de aprendizaje del agente Q-learning")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    save(fig, "10_curva_aprendizaje")

    # Figura 11: ranking del torneo de la mejor configuración
    fig, ax = plt.subplots(figsize=(8, 9.5))
    d = rank_df.iloc[::-1]
    colors = [SERIES[1] if s == best["name"] else SERIES[0] for s in d.strategy]
    ax.barh(d.strategy, d.score, color=colors, height=0.7)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == best["name"]:
            lbl.set_color(SERIES[1]); lbl.set_fontweight("bold")
    ax.set_xlabel("Puntuación media por ronda")
    ax.set_title(f"Torneo con el agente {best['name']}, ruido {best['noise']:g} "
                 f"(posición {best['rank']} de {len(rank_df)})")
    ax.xaxis.grid(True); ax.set_axisbelow(True); ax.margins(y=0.01)
    save(fig, "11_agente_en_ranking")

    # Figura 12: política aprendida (k=1) frente a estrategias de referencia
    states = [START, "CC", "CD", "DC", "DD"]
    cols = ["Ronda 1", "Tras CC", "Tras CD", "Tras DC", "Tras DD"]
    color = {"C": SERIES[2], "D": SERIES[7], "=": AXIS}
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(policy) + 1.4))
    for i, row in enumerate(policy.itertuples(index=False)):
        for j, s in enumerate(states):
            v = getattr(row, s)
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, color=color[v], lw=0))
            ax.text(j + 0.5, i + 0.5, v, ha="center", va="center", color=SURFACE,
                    fontsize=12, fontweight="bold")
    n_agents = sum(policy.policy.str.startswith("Q-learning"))
    ax.axhline(n_agents, color=INK, lw=1)
    ax.set_xlim(0, len(states)); ax.set_ylim(len(policy), 0)
    ax.set_xticks([j + 0.5 for j in range(len(states))], cols)
    ax.set_yticks([i + 0.5 for i in range(len(policy))], policy.policy)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Política aprendida con memoria k=1 frente a estrategias clásicas", pad=28)
    ax.text(0, len(policy) + 0.45, "Estado = (jugada propia, jugada del rival) en la ronda "
            "anterior. Grudger no es de memoria 1: en DC/DD ya guarda rencor.",
            fontsize=8, color=INK2, va="top")
    save(fig, "12_politica_aprendida")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m tfm_axelrod.qlearning")
    p.add_argument("--episodes", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--models-dir", default="models/qlearning")
    p.add_argument("--classifier-dir", default="models/classifier_serving")
    p.add_argument("--results-dir", default="docs/results")
    p.add_argument("--figures-dir", default="docs/figures")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--experiment", default="qlearning_agent")
    args = p.parse_args(argv)
    t_start = time.perf_counter()

    configs = [(k, nz) for nz in NOISES for k in MEMORIES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        base_f = {nz: pool.submit(_baseline_task, nz, args.seed) for nz in NOISES}
        cfg_f = [pool.submit(run_config, k, nz, args.episodes, args.seed) for k, nz in configs]
        bases = {nz: f.result() for nz, f in base_f.items()}
        results = [f.result() for f in cfg_f]
    print(f"Entrenamiento + evaluación (en paralelo): {(time.perf_counter() - t_start) / 60:.1f} min")

    models_dir, results_dir = Path(args.models_dir), Path(args.results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    window = max(1, args.episodes // 50)
    tail = max(1, args.episodes // 10)
    rows, curves, rank_tables = [], [], {}
    for r in results:
        k, nz = r["memory"], r["noise"]
        rk = ranking(bases[nz], r["name"], r["eval"])
        rank_tables[(k, nz)] = rk
        me = rk[rk.strategy == r["name"]].iloc[0]
        beh = behaviour(r["eval"]["matches"])
        pred = classify(r["eval"]["matches"], Path(args.classifier_dir))
        total = sum(pred.values())
        top = pred.most_common(3)
        n_states = sum(4 ** j for j in range(k + 1))
        rows.append({
            "memory": k, "noise": nz, "episodes": args.episodes,
            "final_reward": float(r["curve"][-tail:].mean()),
            "tournament_score": float(me.score), "tournament_rank": int(me["rank"]),
            **beh,
            "states_visited": len(r["q"]), "states_possible": n_states,
            "classifier_top": "; ".join(f"{s} ({c / total:.0%})" for s, c in top),
            "train_seconds": r["train_seconds"],
        })
        r["rank"] = int(me["rank"])
        ma = pd.Series(r["curve"]).rolling(window).mean()
        step = max(1, args.episodes // 400)
        idx = np.arange(window - 1, args.episodes, step)
        curves.append(pd.DataFrame({"config": f"k={k}, ruido {nz:g}", "episode": idx + 1,
                                    "reward_ma": ma.iloc[idx].to_numpy()}))
        q_path = models_dir / f"q_k{k}_noise{nz:g}.json"
        q_path.write_text(json.dumps({
            "memory": k, "noise": nz, "episodes": args.episodes, "seed": args.seed,
            "hyperparams": HYPERPARAMS, "turns": TURNS,
            "q": {s: {"C": v[0], "D": v[1]} for s, v in sorted(r["q"].items())},
            "visits": dict(sorted(r["visits"].items())),
        }, indent=1), encoding="utf-8")
        r["q_path"] = q_path
        r["classifier_counts"] = {s: c / total for s, c in pred.most_common()}

    res = pd.DataFrame(rows)
    res.round(4).to_csv(results_dir / "qlearning_results.csv", index=False)
    ref_rows = [{"strategy": s, "noise": nz, **m}
                for nz, b in bases.items() for s, m in b["reference_behaviour"].items()]
    ref_df = pd.DataFrame(ref_rows)
    ref_df.round(4).to_csv(results_dir / "qlearning_reference_behaviour.csv", index=False)
    policy = k1_policy_table(results)
    policy.to_csv(results_dir / "qlearning_policy_k1.csv", index=False)
    curves_df = pd.concat(curves, ignore_index=True)

    best_i = int(res.sort_values(["tournament_rank", "tournament_score"],
                                 ascending=[True, False]).index[0])
    best = results[best_i]
    make_figures(results, curves_df, best, rank_tables[(best["memory"], best["noise"])],
                 policy, Path(args.figures_dir))

    # --- MLflow (en el proceso principal: sqlite no admite escrituras concurrentes)
    import mlflow

    mlflow.set_tracking_uri(args.tracking_uri)
    exp = mlflow.get_experiment_by_name(args.experiment)
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(
        args.experiment, artifact_location=Path("mlruns").resolve().as_uri())
    for r, row in zip(results, rows):
        with mlflow.start_run(experiment_id=exp_id,
                              run_name=f"qlearning_k{r['memory']}_noise{r['noise']:g}"):
            mlflow.set_tags({"memory": r["memory"], "noise": r["noise"]})
            mlflow.log_params({"memory": r["memory"], "noise": r["noise"],
                               "episodes": args.episodes, "turns": TURNS, "seed": args.seed,
                               "eval_repetitions": EVAL_REPETITIONS, **HYPERPARAMS})
            mlflow.log_metrics({k: v for k, v in row.items()
                                if isinstance(v, (int, float)) and k not in
                                ("memory", "noise", "episodes")})
            ma = pd.Series(r["curve"]).rolling(window).mean()
            for ep in range(window - 1, args.episodes, max(1, args.episodes // 100)):
                mlflow.log_metric("reward_moving_avg", float(ma.iloc[ep]), step=ep + 1)
            mlflow.log_artifact(str(r["q_path"]), artifact_path="q_table")
            mlflow.log_dict(r["classifier_counts"], "classifier_predictions.json")

    elapsed = time.perf_counter() - t_start
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_colwidth", 80):
        print("\nRESULTADOS POR CONFIGURACIÓN\n", res.round(3).to_string(index=False))
        print("\nCOMPORTAMIENTO DE REFERENCIA\n", ref_df.round(3).to_string(index=False))
        print("\nPOLÍTICA k=1\n", policy.fillna("").to_string(index=False))
        for r in results:
            print(f"\nClasificador {r['name']} ruido {r['noise']:g}: " + ", ".join(
                f"{s} {v:.0%}" for s, v in list(r["classifier_counts"].items())[:6]))
    print(f"\nMejor configuración: k={best['memory']}, ruido {best['noise']:g} "
          f"(posición {best['rank']})")
    print(f"Entrenamiento por configuración: "
          + ", ".join(f"k={r['memory']}/ruido {r['noise']:g}: {r['train_seconds']:.0f}s"
                      for r in results))
    print(f"Tiempo total: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
