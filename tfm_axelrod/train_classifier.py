"""Clasificador supervisado: identifica la estrategia a partir de las primeras n rondas.

Uso:
    uv run python -m tfm_axelrod.train_classifier

Para cada n de ``N_VALUES`` entrena regresión logística, Random Forest y
Gradient Boosting (un modelo por n), evalúa en un test separado por partida,
registra todo en MLflow (sqlite local) y guarda el mejor algoritmo en
``models/classifier``. Genera la tabla de métricas y las figuras 07-09.
"""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import skops.io
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tfm_axelrod.features import FEATURE_COLUMNS, META_COLUMNS, N_VALUES

ALGORITHMS: dict[str, dict] = {
    "logistic_regression": {"C": 1.0, "max_iter": 2000},
    "random_forest": {"n_estimators": 200, "min_samples_leaf": 2, "max_features": "sqrt"},
    "gradient_boosting": {"max_iter": 300, "learning_rate": 0.1, "max_leaf_nodes": 31,
                          "early_stopping": True, "validation_fraction": 0.1,
                          "n_iter_no_change": 10},
}
LABELS = {"logistic_regression": "Regresión logística", "random_forest": "Random Forest",
          "gradient_boosting": "Gradient Boosting"}
PAIR = ("Tit For Tat", "Grudger")


def build_model(algo: str, seed: int):
    params = ALGORITHMS[algo]
    if algo == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(**params, random_state=seed))
    if algo == "random_forest":
        return RandomForestClassifier(**params, n_jobs=-1, random_state=seed)
    if algo == "gradient_boosting":
        return HistGradientBoostingClassifier(**params, random_state=seed)
    raise ValueError(algo)


def split_by_match(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.Series, pd.Series]:
    """Máscaras train/test agrupando por partida (todas sus filas y prefijos al mismo lado)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, _ = next(gss.split(df, groups=df.match_key))
    is_train = np.zeros(len(df), dtype=bool)
    is_train[train_idx] = True
    return pd.Series(is_train, index=df.index), pd.Series(~is_train, index=df.index)


def setup_mlflow(tracking_uri: str, experiment: str) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        artifacts = Path("mlruns").resolve().as_uri()
        return mlflow.create_experiment(experiment, artifact_location=artifacts)
    return exp.experiment_id


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m tfm_axelrod.train_classifier")
    p.add_argument("--data", default="data/analytics/classifier_features.parquet")
    p.add_argument("--models-dir", default="models/classifier")
    p.add_argument("--results-dir", default="docs/results")
    p.add_argument("--figures-dir", default="docs/figures")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--experiment", default="strategy_classifier")
    args = p.parse_args(argv)
    t_start = time.perf_counter()

    df = pd.read_parquet(args.data)
    extra = set(df.columns) - set(META_COLUMNS) - set(FEATURE_COLUMNS)
    if extra or not set(FEATURE_COLUMNS) <= set(df.columns):
        raise SystemExit(f"Columnas inesperadas en el dataset: {sorted(extra)}")
    is_train, is_test = split_by_match(df, args.test_size, args.seed)
    classes = sorted(df.player.unique())
    n_train_matches = df[is_train].match_key.nunique()
    n_test_matches = df[is_test].match_key.nunique()
    print(f"Filas: {len(df)} | partidas train/test: {n_train_matches}/{n_test_matches} "
          f"| clases: {len(classes)}")

    exp_id = setup_mlflow(args.tracking_uri, args.experiment)
    pip_reqs = [f"scikit-learn=={version('scikit-learn')}", f"numpy=={version('numpy')}"]

    metrics_rows, preds, fitted, model_uris = [], {}, {}, {}
    for n in N_VALUES:
        d = df[df.n == n]
        tr, te = d[is_train[d.index]], d[is_test[d.index]]
        X_tr, y_tr = tr[FEATURE_COLUMNS].astype(float), tr.player
        X_te, y_te = te[FEATURE_COLUMNS].astype(float), te.player
        for algo in ALGORITHMS:
            model = build_model(algo, args.seed)
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - t0
            proba = model.predict_proba(X_te)
            y_pred = np.asarray(model.classes_)[proba.argmax(axis=1)]
            m = {
                "accuracy": accuracy_score(y_te, y_pred),
                "top3_accuracy": top_k_accuracy_score(y_te, proba, k=3, labels=model.classes_),
                "f1_macro": f1_score(y_te, y_pred, average="macro", labels=classes, zero_division=0),
                "train_seconds": fit_s,
            }
            with mlflow.start_run(experiment_id=exp_id, run_name=f"{algo}_n{n}") as run:
                mlflow.set_tags({"algorithm": algo, "n": n})
                mlflow.log_params({"algorithm": algo, "n": n, "seed": args.seed,
                                   "test_size": args.test_size, "n_train": len(tr),
                                   "n_test": len(te), "n_features": len(FEATURE_COLUMNS),
                                   **ALGORITHMS[algo]})
                mlflow.log_metrics(m)
                # Modelo recién entrenado en este proceso: sus tipos internos son de confianza.
                trusted = skops.io.get_untrusted_types(data=skops.io.dumps(model))
                info = mlflow.sklearn.log_model(model, name="model", pip_requirements=pip_reqs,
                                                input_example=X_tr.head(3),
                                                skops_trusted_types=trusted)
            model_uris[(algo, n)] = info.model_uri
            fitted[(algo, n)] = model
            preds[(algo, n)] = pd.DataFrame({"y": y_te.values, "pred": y_pred,
                                             "noise": te.noise.values})
            metrics_rows.append({"algorithm": algo, "n": n, **m})
            print(f"n={n:>3} {algo:<20} acc={m['accuracy']:.3f} top3={m['top3_accuracy']:.3f} "
                  f"f1={m['f1_macro']:.3f} ({fit_s:.1f}s)")

    metrics = pd.DataFrame(metrics_rows)
    best = metrics.groupby("algorithm").accuracy.mean().idxmax()
    print(f"\nMejor algoritmo (accuracy media sobre n): {best}")

    # --- Registro del mejor algoritmo en el Model Registry (un modelo por n)
    client = MlflowClient()
    for n in N_VALUES:
        name = f"axelrod-strategy-classifier-n{n}"
        mv = mlflow.register_model(model_uris[(best, n)], name)
        client.set_model_version_tag(name, mv.version, "algorithm", best)
        acc = metrics[(metrics.algorithm == best) & (metrics.n == n)].accuracy.item()
        client.set_model_version_tag(name, mv.version, "accuracy", f"{acc:.4f}")

    # --- Modelos del mejor algoritmo + metadata
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    for n in N_VALUES:
        joblib.dump(fitted[(best, n)], models_dir / f"model_n{n}.joblib", compress=3)
    meta = {"algorithm": best, "params": ALGORITHMS[best], "features": FEATURE_COLUMNS,
            "classes": classes, "n_values": N_VALUES, "seed": args.seed,
            "test_size": args.test_size, "split": "GroupShuffleSplit por match_key",
            "sklearn_version": version("scikit-learn")}
    (models_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                              encoding="utf-8")

    # --- Tablas de resultados
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics.round(4).to_csv(results_dir / "classifier_metrics.csv", index=False)

    by_noise = pd.DataFrame([
        {"n": n, "noise": nz, "accuracy": (g.y == g.pred).mean()}
        for n in N_VALUES for nz, g in preds[(best, n)].groupby("noise")
    ])
    by_noise_wide = by_noise.pivot(index="n", columns="noise", values="accuracy")
    by_noise_wide.round(4).to_csv(results_dir / "classifier_accuracy_by_noise.csv")

    pair_rows = []
    for n in N_VALUES:
        g = preds[(best, n)]
        a, b = PAIR
        ga, gb = g[g.y == a], g[g.y == b]
        pair_rows.append({
            "n": n,
            f"{a} -> {a}": (ga.pred == a).mean(), f"{a} -> {b}": (ga.pred == b).mean(),
            f"{b} -> {b}": (gb.pred == b).mean(), f"{b} -> {a}": (gb.pred == a).mean(),
        })
    pair = pd.DataFrame(pair_rows).set_index("n")
    pair.round(4).to_csv(results_dir / "classifier_tft_vs_grudger.csv")

    make_figures(metrics, by_noise, preds[(best, max(N_VALUES))], classes, best,
                 Path(args.figures_dir))

    elapsed = time.perf_counter() - t_start
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nMÉTRICAS EN TEST\n", metrics.round(4).to_string(index=False))
        print(f"\nACCURACY POR RUIDO ({best})\n", by_noise_wide.round(4).to_string())
        print(f"\nCONFUSIÓN {PAIR[0]} / {PAIR[1]} ({best}, proporción de cada clase real)\n",
              pair.round(4).to_string())
    print(f"\nTiempo total: {elapsed / 60:.1f} min")


def make_figures(metrics, by_noise, preds_max_n, classes, best, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    from tfm_axelrod.plot_style import INK, INK2, SEQ, SERIES, SURFACE, apply_style

    apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(fig_dir / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    def line_chart(ax, series: dict[str, pd.Series], title: str, ylabel: str):
        for i, (label, s) in enumerate(series.items()):
            ax.plot(s.index, s.values, color=SERIES[i], lw=2, marker="o", ms=5)
        # Etiquetas al final de cada línea, separadas para que no se solapen.
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.045
        ends = sorted(((s.values[-1], label) for label, s in series.items()), reverse=True)
        prev = None
        for y, label in ends:
            y = y if prev is None else min(y, prev - gap)
            ax.text(max(N_VALUES) + 2, y, label, va="center", fontsize=9, color=INK2)
            prev = y
        ax.set_xticks(N_VALUES)
        ax.set_xlim(0, max(N_VALUES) + 22)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Rondas observadas (n)"); ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.yaxis.grid(True); ax.set_axisbelow(True)

    # Figura 7: accuracy frente a n por algoritmo
    fig, ax = plt.subplots(figsize=(8, 5))
    line_chart(ax, {LABELS[a]: g.set_index("n").accuracy for a, g in metrics.groupby("algorithm")},
               "Accuracy del clasificador según las rondas observadas", "Accuracy en test")
    save(fig, "07_accuracy_vs_rondas")

    # Figura 8: accuracy del mejor algoritmo por nivel de ruido
    fig, ax = plt.subplots(figsize=(8, 5))
    line_chart(ax, {f"Ruido {nz:g}": g.set_index("n").accuracy
                    for nz, g in by_noise.groupby("noise")},
               f"Accuracy por nivel de ruido ({LABELS[best]})", "Accuracy en test")
    save(fig, "08_accuracy_por_ruido")

    # Figura 9: matriz de confusión normalizada por fila, n máximo
    cm = confusion_matrix(preds_max_n.y, preds_max_n.pred, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 10))
    cmap = SEQ.copy()
    cmap.set_bad(SURFACE)  # celdas vacías con el color de fondo
    im = ax.imshow(np.ma.masked_equal(cm, 0), cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] >= 0.05:
                ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=5.5,
                        color="#ffffff" if cm[i, j] > 0.5 else INK)
    ax.set_xlabel("Estrategia predicha"); ax.set_ylabel("Estrategia real")
    ax.set_title(f"Matriz de confusión ({LABELS[best]}, n = {max(N_VALUES)}, % por fila)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Proporción de la clase real",
                 format=PercentFormatter(1.0, decimals=0))
    save(fig, "09_matriz_confusion")


if __name__ == "__main__":
    main()
