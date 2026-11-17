"""Exporta los modelos de un algoritmo ya entrenado desde MLflow para servirlos.

Uso:
    uv run python -m tfm_axelrod.export_serving [--algorithm gradient_boosting]

No reentrena: localiza en MLflow la última ejecución de cada n del algoritmo,
carga su modelo, comprueba que reproduce la accuracy registrada sobre el mismo
test (partición por partida) y lo guarda en ``models/classifier_serving`` con
un ``metadata.json``. Los Gradient Boosting pesan mucho menos que los Random
Forest, por eso son los que usa la API.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import skops.io
from mlflow.tracking import MlflowClient

from tfm_axelrod.features import FEATURE_COLUMNS, N_VALUES
from tfm_axelrod.train_classifier import ALGORITHMS, split_by_match


def load_logged_model(run_id: str):
    """Carga el modelo skops de una ejecución (creado por nosotros: tipos de confianza)."""
    local = Path(mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model"))
    path = next(local.glob("*.skops"))
    return skops.io.load(path, trusted=skops.io.get_untrusted_types(file=path))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m tfm_axelrod.export_serving")
    p.add_argument("--algorithm", default="gradient_boosting", choices=sorted(ALGORITHMS))
    p.add_argument("--data", default="data/analytics/classifier_features.parquet")
    p.add_argument("--out", default="models/classifier_serving")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--experiment", default="strategy_classifier")
    args = p.parse_args(argv)

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        raise SystemExit(f"No existe el experimento {args.experiment!r}: entrena primero")

    df = pd.read_parquet(args.data)
    is_test = None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs_meta, seed, test_size, classes = {}, None, None, None

    for n in N_VALUES:
        runs = client.search_runs(
            [exp.experiment_id],
            filter_string=f"tags.algorithm = '{args.algorithm}' AND tags.n = '{n}'",
            order_by=["attributes.start_time DESC"], max_results=1,
        )
        if not runs:
            raise SystemExit(f"No hay ejecución de {args.algorithm} para n={n}")
        run = runs[0]
        seed, test_size = int(run.data.params["seed"]), float(run.data.params["test_size"])
        if is_test is None:
            _, is_test = split_by_match(df, test_size, seed)
            classes = sorted(df.player.unique())

        model = load_logged_model(run.info.run_id)
        te = df[(df.n == n) & is_test]
        acc = float(np.mean(model.predict(te[FEATURE_COLUMNS].astype(float)) == te.player.values))
        logged = run.data.metrics["accuracy"]
        if abs(acc - logged) > 1e-9:
            raise SystemExit(f"n={n}: accuracy {acc:.6f} != registrada {logged:.6f}")
        if list(model.classes_) != classes:
            raise SystemExit(f"n={n}: clases del modelo distintas de las del dataset")

        joblib.dump(model, out / f"model_n{n}.joblib", compress=3)
        runs_meta[str(n)] = {"mlflow_run_id": run.info.run_id, "accuracy": round(acc, 4),
                             "top3_accuracy": round(run.data.metrics["top3_accuracy"], 4),
                             "f1_macro": round(run.data.metrics["f1_macro"], 4)}
        print(f"n={n:>3} run={run.info.run_id[:8]} accuracy={acc:.4f} (coincide con MLflow)")

    meta = {"algorithm": args.algorithm, "params": ALGORITHMS[args.algorithm],
            "features": FEATURE_COLUMNS, "classes": classes, "n_values": N_VALUES,
            "seed": seed, "test_size": test_size, "split": "GroupShuffleSplit por match_key",
            "sklearn_version": version("scikit-learn"), "test_metrics": runs_meta}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"Guardado en {out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
