"""Comparación de tablas de características (no necesita Spark)."""

from pathlib import Path

import pandas as pd
import pytest

from tfm_axelrod.compare_tables import compare, main, read_table

REFERENCE = Path("data/analytics/classifier_features.parquet")


def make_table(n=4):
    return pd.DataFrame({
        "match_key": [f"ro_x|0.0|100|0|{i}" for i in range(n)],
        "player_index": pd.Series([0, 1] * (n // 2), dtype="int8"),
        "n": pd.Series([5] * n, dtype="int32"),
        "player": ["Tit For Tat"] * n,
        "is_self_play": [False] * n,
        "coop_rate": [0.2, 0.4, 0.6, 0.8],
        "p_coop_after_cd": [0.5, float("nan"), 1.0, 0.0],
    })


def test_identical_and_reordered_tables_match():
    a = make_table()
    b = a.iloc[::-1].reset_index(drop=True)
    b["player_index"] = b["player_index"].astype("int64")  # tipos distintos de la clave
    assert compare(a, b)["match"]


def test_tolerance():
    a, b = make_table(), make_table()
    b.loc[1, "coop_rate"] += 1e-12
    assert compare(a, b, tol=1e-9)["match"]
    b.loc[2, "coop_rate"] += 1e-6
    res = compare(a, b, tol=1e-9)
    assert not res["match"] and any("coop_rate" in p for p in res["problems"])


def test_nan_only_on_one_side_and_text_differences():
    a, b = make_table(), make_table()
    b.loc[1, "p_coop_after_cd"] = 0.5
    b.loc[0, "player"] = "Defector"
    res = compare(a, b)
    assert not res["match"]
    assert any("p_coop_after_cd" in p for p in res["problems"])
    assert any("player" in p for p in res["problems"])


def test_missing_rows_and_duplicates():
    a, b = make_table(), make_table()
    res = compare(a.iloc[:3], b)
    assert not res["match"] and any("filas" in p for p in res["problems"])
    dup = pd.concat([a.iloc[:3], a.iloc[[0]]], ignore_index=True)
    res = compare(dup, b)
    assert any("duplicadas" in p for p in res["problems"])


def test_cli_and_folder_input(tmp_path, capsys):
    a = make_table()
    folder = tmp_path / "cand"
    folder.mkdir()
    a.iloc[:2].to_parquet(folder / "part-0.parquet")
    a.iloc[2:].to_parquet(folder / "part-1.parquet")
    a.to_parquet(tmp_path / "ref.parquet")
    assert len(read_table(folder)) == 4
    main([str(folder), str(tmp_path / "ref.parquet")])
    assert "RESULTADO: COINCIDEN" in capsys.readouterr().out
    b = a.copy()
    b.loc[0, "coop_rate"] = 0.9
    b.to_parquet(tmp_path / "ref2.parquet")
    with pytest.raises(SystemExit) as exc:
        main([str(folder), str(tmp_path / "ref2.parquet")])
    assert exc.value.code == 1 and "NO COINCIDEN" in capsys.readouterr().out
