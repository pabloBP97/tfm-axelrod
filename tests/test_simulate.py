import axelrod as axl
import pandas as pd
import pyarrow.dataset as ds
import pytest

from tfm_axelrod.simulate import run_evolutionary, run_roundrobin
from tfm_axelrod.strategies import FULL, SMALL


def read_rr(out_dir) -> pd.DataFrame:
    """Lee la salida round-robin (noise/turns están en los ficheros y en la ruta)."""
    df = ds.dataset(out_dir / "roundrobin", format="parquet").to_table().to_pandas()
    sort = ["noise", "turns", "match_id", "player_index", "round"]
    return df.sort_values(sort).reset_index(drop=True)


def test_row_count(tmp_path):
    noise, turns, reps = [0.0, 0.05], [10, 20], 3
    _, rows = run_roundrobin(SMALL, noise, turns, reps, seed=1, out_dir=tmp_path)
    n = len(SMALL)
    pairs = n * (n + 1) // 2  # incluye la autopartida
    expected = sum(pairs * 2 * t * reps for t in turns) * len(noise)
    assert rows == expected
    assert len(read_rr(tmp_path)) == expected


def test_tft_vs_defector_no_noise(tmp_path):
    run_roundrobin([axl.TitForTat, axl.Defector], [0.0], [5], 1, seed=0, out_dir=tmp_path)
    df = read_rr(tmp_path)
    m = df[(df.player == "Tit For Tat") & (df.opponent == "Defector")]
    assert m.player_action.tolist() == list("CDDDD")
    assert m.opponent_action.tolist() == list("DDDDD")
    assert m.player_payoff.tolist() == [0, 1, 1, 1, 1]


def test_same_seed_same_result(tmp_path):
    args = (SMALL, [0.0, 0.1], [30], 2)
    id_a, _ = run_roundrobin(*args, seed=7, out_dir=tmp_path / "a")
    id_b, _ = run_roundrobin(*args, seed=7, out_dir=tmp_path / "b")
    id_c, _ = run_roundrobin(*args, seed=8, out_dir=tmp_path / "c")
    a, b, c = (read_rr(tmp_path / x).drop(columns="simulated_at") for x in "abc")
    assert id_a == id_b != id_c
    pd.testing.assert_frame_equal(a, b)
    # Con ruido y Random, otra semilla debe cambiar las jugadas.
    assert not a.player_action.equals(c.player_action)


def test_full_preset():
    assert len(FULL) == 30
    assert len({cls().name for cls in FULL}) == 30


def test_evolutionary(tmp_path):
    _, rows = run_evolutionary(SMALL, [0.0, 0.05], 20, 1, seed=3, out_dir=tmp_path,
                               generations=10)
    df = pd.read_parquet(next((tmp_path / "evolutionary").glob("*.parquet")))
    assert rows == len(df) == 2 * 11 * len(SMALL)  # generación 0..10
    sums = df.groupby(["noise", "generation"]).population_share.sum()
    assert sums.to_numpy() == pytest.approx(1.0)
