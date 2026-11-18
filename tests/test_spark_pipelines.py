"""Pipelines Spark sobre una muestra pequeña de data/raw.

Se omiten solos si falta PySpark (extra ``spark``), Java o la capa raw local.
En Windows sin winutils usan la E/S con pyarrow; la lógica corre en Spark igual.
"""

import os
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyspark")
if not (shutil.which("java") or os.environ.get("JAVA_HOME")):
    pytest.skip("Spark necesita Java y no se encuentra", allow_module_level=True)

from tfm_axelrod import spark_pipelines as sp  # noqa: E402
from tfm_axelrod.compare_tables import compare  # noqa: E402
from tfm_axelrod.features import FEATURE_COLUMNS, compute_features  # noqa: E402

RAW = Path("data/raw")
REFERENCE = Path("data/analytics/classifier_features.parquet")
SAMPLE_PLAYERS = ["Tit For Tat", "Defector", "Cooperator", "Grudger", "Random"]
SAMPLE_PARTS = [("0.0", 100), ("0.05", 100)]

if not (RAW / "roundrobin").exists():
    pytest.skip("No hay capa raw local (data/raw)", allow_module_level=True)


@pytest.fixture(scope="module")
def spark():
    s = sp.build_spark(master="local[2]", driver_memory="2g", shuffle_partitions=4, io="auto")
    yield s
    s.stop()


@pytest.fixture(scope="module")
def sample_tables():
    """Partidas completas entre 5 estrategias, repetición 0, 2 ruidos, 100 rondas."""
    parts = []
    for noise, turns in SAMPLE_PARTS:
        t = ds.dataset(RAW / "roundrobin" / f"noise={noise}" / f"turns={turns}", partitioning=None).to_table()
        df = t.to_pandas()
        keep = df["player"].isin(SAMPLE_PLAYERS) & df["opponent"].isin(SAMPLE_PLAYERS) & (df["repetition"] == 0)
        parts.append(pa.Table.from_pandas(df[keep], schema=t.schema, preserve_index=False))
    evo = pq.read_table(next((RAW / "evolutionary").glob("*.parquet")))
    return parts, evo


def write_raw(root: Path, parts, evo) -> Path:
    raw = root / "raw"
    (raw / "roundrobin").mkdir(parents=True)
    (raw / "evolutionary").mkdir(parents=True)
    for (noise, turns), t in zip(SAMPLE_PARTS, parts):
        # Ficheros planos, como los compactados que hay en raw/ del lake.
        pq.write_table(t, raw / "roundrobin" / f"roundrobin_noise{noise}_turns{turns}.parquet")
    pq.write_table(evo, raw / "evolutionary" / "ev.parquet")
    return raw


@pytest.fixture(scope="module")
def sample_run(spark, sample_tables, tmp_path_factory):
    root = tmp_path_factory.mktemp("spark_sample")
    raw = write_raw(root, *sample_tables)
    result = sp.run_pipelines(spark, raw, root / "out", expected=None, verbose=False)
    return root, result


def read_out(root: Path, table: str) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f) for f in sp.table_path(root / "out", table).rglob("*.parquet")],
                     ignore_index=True)


def test_quality_and_counts(sample_run, sample_tables):
    root, result = sample_run
    parts, evo = sample_tables
    raw_rows = sum(t.num_rows for t in parts)
    n_matches = 15 * len(SAMPLE_PARTS)  # 5 estrategias: 15 pares con autopartida
    assert result["report"]["filas_raw"] == raw_rows == n_matches * 2 * 100
    assert all(result["report"][k] == 0 for k in sp.QUALITY_COLUMNS)
    assert result["bad_generations"] == 0
    assert result["counts"] == {
        "curated_moves": raw_rows,
        "curated_evolution": evo.num_rows,
        "analytics_match_summary": n_matches * 2,
        "analytics_strategy_ranking": 5 * len(SAMPLE_PARTS),
        "analytics_classifier_features": n_matches * 2 * len(sp.N_VALUES),
    }
    for table, layer in sp.TABLES.items():
        assert (root / "out" / layer / table).is_dir()


def test_curated_and_summary_columns(sample_run):
    root, _ = sample_run
    moves = read_out(root, "curated_moves")
    assert list(moves.columns) == [
        "match_key", "tournament_id", "match_id", "repetition", "noise", "turns", "round",
        "player_index", "player", "opponent", "player_action", "opponent_action",
        "player_cooperated", "opponent_cooperated", "player_payoff", "opponent_payoff",
        "is_self_play", "simulated_at", "curated_at"]
    assert moves["match_key"].str.match(r"^ro_[0-9a-f]{12}\|0\.(0|05)\|100\|0\|\d+$").all()
    summary = read_out(root, "analytics_match_summary")
    tft_def = summary[(summary.player == "Tit For Tat") & (summary.opponent == "Defector") & (summary.noise == 0)]
    assert tft_def["player_score"].tolist() == [99] and tft_def["outcome"].tolist() == ["loss"]
    ranking = read_out(root, "analytics_strategy_ranking")
    assert set(ranking.columns) == {"strategy", "noise", "turns", "matches", "avg_score_per_turn",
                                    "avg_cooperation_rate", "win_rate", "rank"}
    assert (ranking["matches"] == 4).all()


def test_features_match_python_reference(sample_run):
    """Las 17 características coinciden con tfm_axelrod.features (implementación independiente)."""
    root, _ = sample_run
    feats = read_out(root, "analytics_classifier_features")
    moves = read_out(root, "curated_moves").sort_values("round")
    assert list(feats.columns) == sp.KEYS + ["n"] + FEATURE_COLUMNS
    seqs = {k: (g["player_action"].tolist(), g["opponent_action"].tolist())
            for k, g in moves.groupby(["match_key", "player_index"])}
    for _, row in feats.sample(60, random_state=0).iterrows():
        own, opp = seqs[(row["match_key"], row["player_index"])]
        ref = compute_features(own[: row["n"]], opp[: row["n"]])
        for col in FEATURE_COLUMNS:
            assert row[col] == pytest.approx(ref[col], abs=1e-9), (col, row["match_key"], row["n"])


@pytest.mark.skipif(not REFERENCE.exists(), reason="Falta la tabla de Databricks")
def test_features_match_databricks(sample_run):
    root, _ = sample_run
    feats = read_out(root, "analytics_classifier_features")
    ref = pd.read_parquet(REFERENCE)
    ref = ref[ref["match_key"].isin(set(feats["match_key"]))]
    assert len(ref) == len(feats)
    res = compare(feats, ref, tol=1e-9)
    assert res["match"], res["problems"]


def test_invalid_rows_are_reported_and_fail(spark, sample_tables, tmp_path):
    parts, evo = sample_tables
    bad = parts[0].to_pandas()
    bad.loc[0, "player_action"] = "X"            # acción inválida
    bad.loc[1, "player_payoff"] = 4              # pago incoherente
    bad = bad.drop(index=[2])                    # deja una partida incompleta
    parts = [pa.Table.from_pandas(bad, schema=parts[0].schema, preserve_index=False), parts[1]]
    raw = write_raw(tmp_path, parts, evo)
    with pytest.raises(sp.PipelineError) as exc:
        sp.run_pipelines(spark, raw, tmp_path / "out", expected=None, verbose=False)
    msg = str(exc.value)
    # La fila con "X" tampoco tiene pago esperado, así que cuenta también como pago incoherente.
    assert "acciones_invalidas=1" in msg and "pagos_incoherentes=2" in msg and "partidas_incompletas" in msg
    # Las filas malas no pasan a curated (la partida incompleta entera se descarta).
    moves = read_out(tmp_path, "curated_moves")
    assert len(moves) == sum(t.num_rows for t in sample_tables[0]) - 200


def test_unexpected_counts_fail(sample_run):
    _, result = sample_run
    sp.validate(result, None)
    sp.validate(result, dict(result["counts"]))
    with pytest.raises(sp.PipelineError, match="curated_moves"):
        sp.validate(result, {**result["counts"], "curated_moves": sp.EXPECTED_COUNTS["curated_moves"]})


def test_cli_exit_code(spark, sample_tables, tmp_path, monkeypatch, capsys):
    raw = write_raw(tmp_path, *sample_tables)
    monkeypatch.setattr(sp, "build_spark", lambda *a, **k: spark)
    monkeypatch.setattr(spark, "stop", lambda: None)  # la sesión es del fixture
    sp.main(["--raw", str(raw), "--out", str(tmp_path / "ok"), "--no-check-counts"])
    assert "RESUMEN validacion=OK" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:  # muestra != recuentos del conjunto completo
        sp.main(["--raw", str(raw), "--out", str(tmp_path / "ko")])
    assert exc.value.code != 0 and "se esperaban 11,160,000" in str(exc.value.code)


# --------------------------------------------------------------------------- #
# Modo --from-lake con un cliente de Azure simulado
# --------------------------------------------------------------------------- #
class FakePath:
    def __init__(self, name, is_directory=False):
        self.name, self.is_directory = name, is_directory


class FakeDownload:
    def __init__(self, data):
        self.data = data

    def readinto(self, fh):
        fh.write(self.data)
        return len(self.data)


class FakeFile:
    def __init__(self, fs, path):
        self.fs, self.path = fs, path

    def upload_data(self, data, length=None, overwrite=False):
        content = data.read()
        assert length == len(content) and overwrite
        self.fs.files[self.path] = content
        self.fs.uploaded.append(self.path)

    def download_file(self):
        self.fs.downloaded.append(self.path)
        return FakeDownload(self.fs.files[self.path])


class FakeDirectory:
    def __init__(self, fs, path):
        self.fs, self.path = fs, path

    def exists(self):
        return any(p.startswith(self.path + "/") for p in self.fs.files)

    def delete_directory(self):
        self.fs.deleted.append(self.path)
        for p in [p for p in self.fs.files if p.startswith(self.path + "/")]:
            del self.fs.files[p]


class FakeLake:
    file_system_name = "lake"

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.uploaded, self.downloaded, self.deleted = [], [], []

    def exists(self):
        return True

    def get_paths(self, path, recursive=True):
        names = [p for p in self.files if p.startswith(path.rstrip("/") + "/")]
        return [FakePath(n) for n in names]

    def get_file_client(self, path):
        return FakeFile(self, path)

    def get_directory_client(self, path):
        return FakeDirectory(self, path)


def lake_with_raw(tmp_path, parts, evo) -> FakeLake:
    raw = write_raw(tmp_path / "src", parts, evo)
    lake = FakeLake()
    for f in raw.rglob("*.parquet"):
        lake.files[f"raw/{f.relative_to(raw).as_posix()}"] = f.read_bytes()
        lake.files[f"raw_cloud/{f.relative_to(raw).as_posix()}"] = f.read_bytes()  # duplicado: no debe leerse
    lake.files["curated/curated_moves/part-viejo.parquet"] = b"de una ejecucion anterior"
    return lake


def test_from_lake_downloads_raw_only_and_uploads_tables(spark, sample_tables, tmp_path):
    lake = lake_with_raw(tmp_path, *sample_tables)
    result = sp.run_from_lake(lake, spark, tmp_path / "work", expected=None)
    assert sorted(lake.downloaded) == sorted(p for p in lake.files if p.startswith("raw/"))
    assert not any(p.startswith("raw_cloud/") for p in lake.downloaded)
    assert "curated/curated_moves" in lake.deleted
    assert "curated/curated_moves/part-viejo.parquet" not in lake.files
    assert lake.uploaded and all(p.endswith(".parquet") for p in lake.uploaded)
    assert {p.split("/")[0] + "/" + p.split("/")[1] for p in lake.uploaded} == \
        {f"{layer}/{table}" for table, layer in sp.TABLES.items()}
    assert result["uploaded_files"] == len(lake.uploaded)
    # Lo subido es legible y tiene las filas esperadas.
    feats = [k for k in lake.uploaded if k.startswith("analytics/analytics_classifier_features/")]
    n = 0
    for k in feats:
        f = tmp_path / "check" / k
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(lake.files[k])
        n += pq.read_metadata(f).num_rows
    assert n == result["counts"]["analytics_classifier_features"]


def test_from_lake_uploads_nothing_if_validation_fails(spark, sample_tables, tmp_path):
    lake = lake_with_raw(tmp_path, *sample_tables)
    with pytest.raises(sp.PipelineError):
        sp.run_from_lake(lake, spark, tmp_path / "work", expected=sp.EXPECTED_COUNTS)
    assert lake.uploaded == [] and lake.deleted == []
    assert "curated/curated_moves/part-viejo.parquet" in lake.files


def test_from_lake_fails_if_raw_is_empty(spark, tmp_path):
    lake = FakeLake()
    with pytest.raises(sp.PipelineError, match="No hay ficheros Parquet"):
        sp.run_from_lake(lake, spark, tmp_path / "work", expected=None)
