import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from tfm_axelrod.features import FEATURE_COLUMNS, N_VALUES, compute_features

SERVING = Path("models/classifier_serving")


@pytest.fixture(scope="module")
def meta():
    if not (SERVING / "metadata.json").exists():
        pytest.skip("Falta models/classifier_serving")
    return json.loads((SERVING / "metadata.json").read_text(encoding="utf-8"))


def test_metadata(meta):
    assert meta["algorithm"] == "gradient_boosting"
    assert meta["features"] == FEATURE_COLUMNS
    assert meta["n_values"] == N_VALUES
    assert len(meta["classes"]) == 30
    for n in N_VALUES:
        assert (SERVING / f"model_n{n}.joblib").exists()


@pytest.mark.parametrize("n", [5, 100])
def test_predict_from_actions(meta, n):
    """Flujo de la API: jugadas -> compute_features -> modelo -> probabilidades."""
    model = joblib.load(SERVING / f"model_n{n}.joblib")
    assert list(model.classes_) == meta["classes"]
    # Alternar C/D frente a un cooperador solo lo hace Alternator (frente a un
    # Defector, Win-Stay Lose-Shift jugaría exactamente lo mismo).
    own, opp = (["C", "D"] * n)[:n], ["C"] * n
    X = pd.DataFrame([compute_features(own, opp)])[meta["features"]]
    proba = model.predict_proba(X)[0]
    assert proba.sum() == pytest.approx(1.0)
    assert model.classes_[proba.argmax()] == "Alternator"
