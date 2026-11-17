from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pytest

from tfm_axelrod.features import FEATURE_COLUMNS, META_COLUMNS, N_VALUES, compute_features

RAW = Path("data/raw/roundrobin")
FEATURES = Path("data/analytics/classifier_features.parquet")


def test_tft_vs_defector():
    f = compute_features(list("CDDDD"), list("DDDDD"))
    assert list(f) == FEATURE_COLUMNS
    assert f["coop_rate"] == 0.2 and f["opp_coop_rate"] == 0.0
    assert f["first_move_coop"] == 1
    assert f["avg_payoff"] == pytest.approx(4 / 5)
    assert f["switch_rate"] == pytest.approx(1 / 4)
    assert f["max_c_streak"] == 0.2 and f["max_d_streak"] == 0.8
    assert f["first_defection"] == pytest.approx(2 / 5)
    assert f["state_entropy"] == pytest.approx(-(0.2 * np.log2(0.2) + 0.8 * np.log2(0.8)))
    # Estado previo cd ocurre 1 vez (ronda 1) y tras él deserta; dd ocurre 3 veces.
    assert f["p_coop_after_cd"] == 0.0 and f["freq_after_cd"] == pytest.approx(1 / 4)
    assert f["p_coop_after_dd"] == 0.0 and f["freq_after_dd"] == pytest.approx(3 / 4)
    assert f["p_coop_after_cc"] == 0.5 and f["freq_after_cc"] == 0.0


def test_never_defects():
    f = compute_features(list("CCC"), list("CDC"))
    assert f["first_defection"] == pytest.approx(4 / 3)
    assert f["max_d_streak"] == 0.0
    assert f["avg_payoff"] == pytest.approx((3 + 0 + 3) / 3)


def test_invalid_input():
    with pytest.raises(ValueError):
        compute_features(list("CC"), list("C"))
    with pytest.raises(ValueError):
        compute_features(list("CX"), list("CC"))
    with pytest.raises(ValueError):
        compute_features([], [])


@pytest.mark.skipif(not (RAW.exists() and FEATURES.exists()),
                    reason="Faltan data/raw o data/analytics (no están en git)")
def test_matches_spark_features():
    """Recalcula las características de partidas reales y las compara con Spark."""
    feats = pd.read_parquet(FEATURES)
    keys = feats.drop_duplicates("match_key").groupby("noise").sample(6, random_state=0).match_key
    sample = feats[feats.match_key.isin(keys)]
    parts = sample.match_key.str.split("|", expand=True)
    match_ids = sorted(parts[4].astype(int).unique().tolist())

    raw = (ds.dataset(RAW, format="parquet")
           .to_table(columns=["tournament_id", "noise", "turns", "repetition", "match_id",
                              "player_index", "round", "player_action", "opponent_action"],
                     filter=pc.field("match_id").isin(match_ids))
           .to_pandas())
    raw["match_key"] = (raw.tournament_id + "|" + raw.noise.map(repr) + "|"
                        + raw.turns.astype(str) + "|" + raw.repetition.astype(str) + "|"
                        + raw.match_id.astype(str))
    raw = raw.sort_values(["match_key", "player_index", "round"])
    moves = {k: g for k, g in raw.groupby(["match_key", "player_index"])}

    assert set(sample.match_key) <= {k for k, _ in moves}
    assert set(sample.n) == set(N_VALUES)
    assert set(FEATURE_COLUMNS) == set(feats.columns) - set(META_COLUMNS)
    checked = 0
    for row in sample.itertuples(index=False):
        g = moves[(row.match_key, row.player_index)]
        own = g.player_action.tolist()[: row.n]
        opp = g.opponent_action.tolist()[: row.n]
        got = compute_features(own, opp)
        for col in FEATURE_COLUMNS:
            assert got[col] == pytest.approx(getattr(row, col), abs=1e-9), (row.match_key, row.n, col)
        checked += 1
    assert checked == len(sample) == 4 * 6 * 2 * len(N_VALUES)
