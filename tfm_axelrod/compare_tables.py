"""Compara dos versiones de analytics_classifier_features (p. ej. Spark en ACI vs Databricks).

Coinciden si tienen el mismo número de filas, las mismas claves
(match_key, player_index, n) sin duplicados, y los mismos valores por clave:
las columnas numéricas con tolerancia absoluta (1e-9 por defecto) y el resto
exactamente.

    uv run python -m tfm_axelrod.compare_tables <candidata> data/analytics/classifier_features.parquet

``<candidata>`` puede ser un fichero Parquet o una carpeta con varios. Termina
con código 0 si coinciden y 1 si no.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["match_key", "player_index", "n"]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No hay ficheros Parquet en {path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def compare(candidate: pd.DataFrame, reference: pd.DataFrame, key=KEY, tol: float = 1e-9) -> dict:
    """Devuelve ``{"match": bool, "problems": [...], "max_abs_diff": {col: diff}}``."""
    problems: list[str] = []
    max_diff: dict[str, float] = {}
    if len(candidate) != len(reference):
        problems.append(f"filas: candidata {len(candidate):,} vs referencia {len(reference):,}")
    missing = [c for c in reference.columns if c not in candidate.columns]
    extra = [c for c in candidate.columns if c not in reference.columns]
    if missing:
        problems.append(f"faltan columnas: {missing}")
    if extra:
        problems.append(f"columnas de más: {extra}")
    for name, df in (("candidata", candidate), ("referencia", reference)):
        dup = int(df.duplicated(key).sum())
        if dup:
            problems.append(f"{dup:,} claves duplicadas en la {name}")

    cand = candidate.copy()
    ref = reference.copy()
    for df in (cand, ref):  # tipos de clave homogéneos (int8/int32 vs int64)
        df["player_index"] = df["player_index"].astype("int64")
        df["n"] = df["n"].astype("int64")
        df["match_key"] = df["match_key"].astype(str)
    merged = cand.merge(ref, on=key, how="outer", suffixes=("_c", "_r"), indicator=True)
    only_c = int((merged["_merge"] == "left_only").sum())
    only_r = int((merged["_merge"] == "right_only").sum())
    if only_c or only_r:
        problems.append(f"claves solo en la candidata: {only_c:,}; solo en la referencia: {only_r:,}")
    both = merged[merged["_merge"] == "both"]

    for col in [c for c in reference.columns if c not in key and c in candidate.columns]:
        a, b = both[f"{col}_c"], both[f"{col}_r"]
        if pd.api.types.is_bool_dtype(b) or not pd.api.types.is_numeric_dtype(b):
            bad = int((a.astype(str) != b.astype(str)).sum())
            max_diff[col] = float(bad)
            if bad:
                problems.append(f"{col}: {bad:,} valores distintos")
            continue
        x, y = a.astype("float64").to_numpy(), b.astype("float64").to_numpy()
        both_nan = np.isnan(x) & np.isnan(y)
        diff = np.where(both_nan, 0.0, np.abs(x - y))
        diff = np.where(np.isnan(diff), np.inf, diff)  # NaN solo en un lado
        max_diff[col] = float(diff.max()) if len(diff) else 0.0
        bad = int((diff > tol).sum())
        if bad:
            problems.append(f"{col}: {bad:,} valores difieren más de {tol:g} (máx {max_diff[col]:.3g})")

    return {"match": not problems, "problems": problems, "max_abs_diff": max_diff,
            "rows_candidate": len(candidate), "rows_reference": len(reference)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m tfm_axelrod.compare_tables", description=__doc__.split("\n")[0])
    p.add_argument("candidate")
    p.add_argument("reference")
    p.add_argument("--tol", type=float, default=1e-9)
    args = p.parse_args(argv)

    res = compare(read_table(args.candidate), read_table(args.reference), tol=args.tol)
    print(f"Filas: candidata {res['rows_candidate']:,} | referencia {res['rows_reference']:,}")
    worst = max(res["max_abs_diff"].values(), default=0.0)
    print(f"Mayor diferencia absoluta en columnas numéricas: {worst:.3g} (tolerancia {args.tol:g})")
    if res["match"]:
        print("RESULTADO: COINCIDEN")
        return
    print("RESULTADO: NO COINCIDEN")
    for prob in res["problems"]:
        print(f"  - {prob}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
