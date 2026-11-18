"""Pipelines de Spark raw -> curated -> analytics fuera de Databricks.

Porta, sin cambiar reglas ni definiciones, la lógica de los notebooks
``pipelines/01_raw_to_curated.py`` y ``pipelines/02_curated_to_analytics.py``.
Las tablas de Unity Catalog se sustituyen por carpetas Parquet:

    <out>/curated/curated_moves
    <out>/curated/curated_evolution
    <out>/analytics/analytics_match_summary
    <out>/analytics/analytics_strategy_ranking
    <out>/analytics/analytics_classifier_features

Igual que en los notebooks, cada tabla se escribe y se vuelve a leer antes de
usarla en la siguiente (``spark.table(...)``).

Uso local (la entrada es la capa raw con ``roundrobin/`` y ``evolutionary/``):

    uv run --extra spark python -m tfm_axelrod.spark_pipelines --raw data/raw --out data/spark

Uso en la nube (descarga ``raw/`` del lake, ejecuta y sube ``curated/`` y
``analytics/``; requiere AZURE_STORAGE_CONNECTION_STRING):

    python -m tfm_axelrod.spark_pipelines --from-lake

Termina con código distinto de 0 si hay filas inválidas o si los recuentos
no son los esperados (``EXPECTED_COUNTS``, desactivable con
``--no-check-counts`` para muestras).

PySpark es una dependencia opcional (extra ``spark``).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from tfm_axelrod.cloud import (UploadError, check_filesystem, delete_directory, download_prefix,
                               get_filesystem_client, upload_files)

TABLES = {
    "curated_moves": "curated",
    "curated_evolution": "curated",
    "analytics_match_summary": "analytics",
    "analytics_strategy_ranking": "analytics",
    "analytics_classifier_features": "analytics",
}

EXPECTED_COUNTS = {
    "curated_moves": 11_160_000,
    "curated_evolution": 12_120,
    "analytics_match_summary": 74_400,
    "analytics_strategy_ranking": 240,
    "analytics_classifier_features": 520_800,
}

QUALITY_COLUMNS = ["acciones_invalidas", "rondas_fuera_de_rango", "pagos_incoherentes",
                   "partidas_incompletas"]

# Mismas constantes que el notebook 02 (y que tfm_axelrod.features).
N_VALUES = [5, 10, 20, 30, 50, 75, 100]
STATES = {"cc": (1, 1), "cd": (1, 0), "dc": (0, 1), "dd": (0, 0)}  # (jugada propia, jugada del rival)
KEYS = ["match_key", "player_index", "player", "opponent", "noise", "turns", "repetition", "is_self_play"]


class PipelineError(RuntimeError):
    """Fallo de validación de los pipelines (filas inválidas o recuentos)."""


# --------------------------------------------------------------------------- #
# Sesión de Spark y E/S
# --------------------------------------------------------------------------- #
def build_spark(master: str | None = None, driver_memory: str | None = None,
                shuffle_partitions: int | None = None, io: str = "spark") -> SparkSession:
    """SparkSession local. Valores por defecto pensados para 2 vCPU y 8 GB.

    En modo local el driver es también el ejecutor, así que toda la memoria de
    Spark es ``spark.driver.memory``; se dejan ~3 GB para el sistema, Python y
    la memoria fuera del heap de la JVM. Con ``io="arrow"`` (solo Windows) las
    tablas se recogen enteras en el driver, así que se quita el límite
    ``spark.driver.maxResultSize``.
    """
    # Los workers de Python (p. ej. createDataFrame desde una lista) usan el mismo intérprete.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    master = master or os.environ.get("SPARK_MASTER", "local[2]")
    driver_memory = driver_memory or os.environ.get("SPARK_DRIVER_MEMORY", "5g")
    shuffle_partitions = shuffle_partitions or int(os.environ.get("SPARK_SHUFFLE_PARTITIONS", "16"))
    builder = (
        SparkSession.builder.appName("tfm-axelrod-pipelines")
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
    )
    if resolve_io(io) == "arrow":
        builder = builder.config("spark.driver.maxResultSize", "0")
    if os.environ.get("SPARK_LOCAL_DIRS"):
        builder = builder.config("spark.local.dir", os.environ["SPARK_LOCAL_DIRS"])
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


class SparkParquetIO:
    """E/S nativa de Spark (la que se usa en el contenedor Linux)."""

    name = "spark"

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def read(self, path: str | Path) -> DataFrame:
        # recursiveFileLookup: lee todos los Parquet bajo la carpeta sin inferir
        # particiones de las rutas; noise y turns ya son columnas de los ficheros
        # (igual que los Parquet compactados que se leyeron en Databricks).
        return self.spark.read.option("recursiveFileLookup", "true").parquet(str(path))

    def write(self, df: DataFrame, path: str | Path) -> None:
        df.write.mode("overwrite").parquet(str(path))


class ArrowParquetIO:
    """E/S con pyarrow para Windows sin los binarios nativos de Hadoop (winutils).

    Solo cambia la lectura y escritura de ficheros: todas las transformaciones
    se siguen ejecutando en Spark.
    """

    name = "arrow"

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def read(self, path: str | Path) -> DataFrame:
        import pyarrow.dataset as ds

        files = sorted(str(p) for p in Path(path).rglob("*.parquet"))
        if not files:
            raise PipelineError(f"No hay ficheros Parquet en {path}")
        return self.spark.createDataFrame(ds.dataset(files, format="parquet").to_table())

    def write(self, df: DataFrame, path: str | Path) -> None:
        import pyarrow.parquet as pq

        path = Path(path)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        pq.write_table(df.toArrow(), path / "part-00000.parquet", compression="snappy")


def resolve_io(kind: str) -> str:
    if kind == "auto":
        return "arrow" if os.name == "nt" and not os.environ.get("HADOOP_HOME") else "spark"
    return kind


def make_io(kind: str, spark: SparkSession):
    return {"spark": SparkParquetIO, "arrow": ArrowParquetIO}[resolve_io(kind)](spark)


# --------------------------------------------------------------------------- #
# Notebook 01: raw -> curated
# --------------------------------------------------------------------------- #
def check_moves(raw: DataFrame) -> DataFrame:
    """Añade match_key y las columnas de validación (celda 1 del notebook 01)."""
    match_key = F.concat_ws(
        "|",
        F.col("tournament_id"),
        F.col("noise").cast("string"),
        F.col("turns").cast("string"),
        F.col("repetition").cast("string"),
        F.col("match_id").cast("string"),
    )
    # Pago esperado según la matriz estándar (R=3, S=0, T=5, P=1)
    pa, oa = F.col("player_action"), F.col("opponent_action")
    expected_payoff = (
        F.when((pa == "C") & (oa == "C"), 3)
        .when((pa == "C") & (oa == "D"), 0)
        .when((pa == "D") & (oa == "C"), 5)
        .when((pa == "D") & (oa == "D"), 1)
    )
    return (
        raw.withColumn("match_key", match_key)
        .withColumn("ok_actions", F.coalesce(pa.isin("C", "D") & oa.isin("C", "D"), F.lit(False)))
        .withColumn("ok_round", F.coalesce(F.col("round").between(1, F.col("turns")), F.lit(False)))
        .withColumn("ok_payoff", F.coalesce(F.col("player_payoff") == expected_payoff, F.lit(False)))
        .withColumn("rows_in_round", F.count("*").over(Window.partitionBy("match_key", "round")))
        .withColumn("rows_in_match", F.count("*").over(Window.partitionBy("match_key")))
        .withColumn("ok_shape", (F.col("rows_in_round") == 2) & (F.col("rows_in_match") == 2 * F.col("turns")))
    )


def quality_report(checked: DataFrame) -> dict[str, int]:
    """Informe de calidad (celda 2 del notebook 01) como diccionario."""
    row = checked.agg(
        F.count("*").alias("filas_raw"),
        F.sum((~F.col("ok_actions")).cast("int")).alias("acciones_invalidas"),
        F.sum((~F.col("ok_round")).cast("int")).alias("rondas_fuera_de_rango"),
        F.sum((~F.col("ok_payoff")).cast("int")).alias("pagos_incoherentes"),
        F.sum((~F.col("ok_shape")).cast("int")).alias("partidas_incompletas"),
    ).first()
    return {k: int(v or 0) for k, v in row.asDict().items()}


def curate_moves(checked: DataFrame) -> DataFrame:
    pa, oa = F.col("player_action"), F.col("opponent_action")
    return (
        checked.filter("ok_actions AND ok_round AND ok_payoff AND ok_shape")
        .select(
            "match_key", "tournament_id", "match_id", "repetition", "noise", "turns",
            "round", "player_index", "player", "opponent",
            "player_action", "opponent_action",
            (pa == "C").alias("player_cooperated"),
            (oa == "C").alias("opponent_cooperated"),
            F.col("player_payoff").cast("int").alias("player_payoff"),
            F.col("opponent_payoff").cast("int").alias("opponent_payoff"),
            (F.col("player") == F.col("opponent")).alias("is_self_play"),
            "simulated_at",
            F.current_timestamp().alias("curated_at"),
        )
    )


def evolution_bad_generations(evo: DataFrame) -> int:
    """Generaciones cuya población no suma 1 (celda 3 del notebook 01)."""
    sums = evo.groupBy("tournament_id", "noise", "generation").agg(F.sum("population_share").alias("total"))
    return sums.filter(~F.col("total").between(0.999, 1.001)).count()


def curate_evolution(evo: DataFrame) -> DataFrame:
    return evo.withColumn("curated_at", F.current_timestamp())


# --------------------------------------------------------------------------- #
# Notebook 02: curated -> analytics
# --------------------------------------------------------------------------- #
def match_summary(moves: DataFrame) -> DataFrame:
    return (
        moves.groupBy("match_key", "tournament_id", "noise", "turns", "repetition",
                      "player_index", "player", "opponent", "is_self_play")
        .agg(
            F.sum("player_payoff").alias("player_score"),
            F.sum("opponent_payoff").alias("opponent_score"),
            F.avg(F.col("player_cooperated").cast("int")).alias("cooperation_rate"),
            F.avg(F.col("opponent_cooperated").cast("int")).alias("opponent_cooperation_rate"),
        )
        .withColumn("score_per_turn", F.col("player_score") / F.col("turns"))
        .withColumn(
            "outcome",
            F.when(F.col("player_score") > F.col("opponent_score"), "win")
            .when(F.col("player_score") < F.col("opponent_score"), "loss")
            .otherwise("draw"),
        )
    )


def strategy_ranking(summary: DataFrame) -> DataFrame:
    return (
        summary.filter(~F.col("is_self_play"))
        .groupBy(F.col("player").alias("strategy"), "noise", "turns")
        .agg(
            F.count("*").alias("matches"),
            F.avg("score_per_turn").alias("avg_score_per_turn"),
            F.avg("cooperation_rate").alias("avg_cooperation_rate"),
            F.avg((F.col("outcome") == "win").cast("int")).alias("win_rate"),
        )
        .withColumn("rank", F.rank().over(
            Window.partitionBy("noise", "turns").orderBy(F.desc("avg_score_per_turn"))))
    )


def classifier_features(moves: DataFrame, spark: SparkSession) -> DataFrame:
    """Las 17 características por (partida, jugador, n) del notebook 02."""
    w = Window.partitionBy("match_key", "player_index").orderBy("round")
    seq = (
        moves
        .withColumn("c", F.col("player_cooperated").cast("int"))
        .withColumn("oc", F.col("opponent_cooperated").cast("int"))
        .withColumn("prev_c", F.lag("c").over(w))
        .withColumn("prev_oc", F.lag("oc").over(w))
        .withColumn("switched", F.when(F.col("prev_c").isNull(), 0)
                    .otherwise((F.col("c") != F.col("prev_c")).cast("int")))
        .withColumn("run_id", F.sum("switched").over(w))
        .withColumn("streak_len", F.row_number().over(
            Window.partitionBy("match_key", "player_index", "run_id").orderBy("round")))
    )

    ns = spark.createDataFrame([(n,) for n in N_VALUES], "n int")
    prefix = seq.join(F.broadcast(ns), F.col("round") <= F.col("n"))

    def cnt(cond):
        return F.sum(F.when(cond, 1).otherwise(0))

    aggs = [
        F.avg("c").alias("coop_rate"),
        F.avg("oc").alias("opp_coop_rate"),
        F.max(F.when(F.col("round") == 1, F.col("c"))).alias("first_move_coop"),
        F.avg("player_payoff").alias("avg_payoff"),
        F.sum("switched").alias("n_switches"),
        F.max(F.when(F.col("c") == 1, F.col("streak_len"))).alias("max_c_streak_raw"),
        F.max(F.when(F.col("c") == 0, F.col("streak_len"))).alias("max_d_streak_raw"),
        F.min(F.when(F.col("c") == 0, F.col("round"))).alias("first_defection_raw"),
    ]
    for name, (own, opp) in STATES.items():
        after = (F.col("prev_c") == own) & (F.col("prev_oc") == opp)
        aggs += [
            cnt(after).alias(f"n_after_{name}"),
            cnt(after & (F.col("c") == 1)).alias(f"coop_after_{name}"),
            cnt((F.col("c") == own) & (F.col("oc") == opp)).alias(f"n_now_{name}"),
        ]

    feats = prefix.groupBy(*KEYS, "n").agg(*aggs)

    entropy = F.lit(0.0)
    for name in STATES:
        p = F.col(f"n_now_{name}") / F.col("n")
        entropy = entropy + F.when(p > 0, -p * F.log2(p)).otherwise(0.0)
        feats = (
            feats
            .withColumn(f"p_coop_after_{name}", F.coalesce(
                F.try_divide(F.col(f"coop_after_{name}"), F.col(f"n_after_{name}")), F.lit(0.5)))
            .withColumn(f"freq_after_{name}", F.col(f"n_after_{name}") / (F.col("n") - 1))
        )

    return (
        feats
        .withColumn("state_entropy", entropy)
        .withColumn("switch_rate", F.col("n_switches") / (F.col("n") - 1))
        .withColumn("max_c_streak", F.coalesce(F.col("max_c_streak_raw"), F.lit(0)) / F.col("n"))
        .withColumn("max_d_streak", F.coalesce(F.col("max_d_streak_raw"), F.lit(0)) / F.col("n"))
        .withColumn("first_defection", F.coalesce(F.col("first_defection_raw"), F.col("n") + 1) / F.col("n"))
        .select(
            *KEYS, "n", "coop_rate", "opp_coop_rate", "first_move_coop", "avg_payoff",
            "switch_rate", "max_c_streak", "max_d_streak", "first_defection", "state_entropy",
            *[f"p_coop_after_{s}" for s in STATES], *[f"freq_after_{s}" for s in STATES],
        )
    )


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #
def table_path(out_dir: str | Path, table: str) -> Path:
    return Path(out_dir) / TABLES[table] / table


def run_pipelines(spark: SparkSession, raw_dir: str | Path, out_dir: str | Path, io=None,
                  expected: dict[str, int] | None = EXPECTED_COUNTS, verbose: bool = True) -> dict:
    """Ejecuta los dos notebooks. Devuelve el informe y los recuentos.

    Lanza ``PipelineError`` si hay filas inválidas (también generaciones
    evolutivas que no suman 1) o si los recuentos no coinciden con
    ``expected`` (``None`` desactiva esa comprobación).
    """
    io = io or make_io("auto", spark)
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    say = print if verbose else (lambda *a, **k: None)
    counts: dict[str, int] = {}

    def save_and_reload(df: DataFrame, table: str) -> DataFrame:
        t0 = time.time()
        path = table_path(out_dir, table)
        io.write(df, path)
        reloaded = io.read(path)
        counts[table] = reloaded.count()
        say(f"Filas en {table}: {counts[table]:,} ({time.time() - t0:.0f} s)", flush=True)
        return reloaded

    # ---- Notebook 01 ----------------------------------------------------- #
    checked = check_moves(io.read(raw_dir / "roundrobin"))
    report = quality_report(checked)
    say("Informe de calidad de raw/roundrobin:", flush=True)
    for k, v in report.items():
        say(f"  {k:<24}{v:>14,}", flush=True)
    moves = save_and_reload(curate_moves(checked), "curated_moves")

    evo = io.read(raw_dir / "evolutionary")
    bad_generations = evolution_bad_generations(evo)
    say(f"Generaciones cuya población no suma 1: {bad_generations}", flush=True)
    evolution = save_and_reload(curate_evolution(evo), "curated_evolution")
    if verbose:
        evolution.groupBy("noise").agg(F.countDistinct("generation").alias("generaciones")).orderBy("noise").show()

    # ---- Notebook 02 ----------------------------------------------------- #
    summary = save_and_reload(match_summary(moves), "analytics_match_summary")
    ranking = save_and_reload(strategy_ranking(summary), "analytics_strategy_ranking")
    if verbose:
        ranking.filter("noise = 0 AND turns = 200 AND rank <= 5").orderBy("rank").show(truncate=False)
    features = save_and_reload(classifier_features(moves, spark), "analytics_classifier_features")
    if verbose:
        (features.filter("player = 'Tit For Tat' AND opponent = 'Defector' AND noise = 0 "
                         "AND turns = 100 AND repetition = 0 AND n = 10")
         .select("n", "coop_rate", "first_move_coop", "first_defection", "max_d_streak", "p_coop_after_dd")
         .show())

    result = {"report": report, "bad_generations": bad_generations, "counts": counts}
    validate(result, expected)
    return result


def validate(result: dict, expected: dict[str, int] | None) -> None:
    problems = [f"{k}={result['report'][k]:,}" for k in QUALITY_COLUMNS if result["report"][k] > 0]
    if result["bad_generations"] > 0:
        problems.append(f"generaciones_que_no_suman_1={result['bad_generations']}")
    if expected:
        for table, n in expected.items():
            got = result["counts"].get(table)
            if got != n:
                problems.append(f"{table}: {got:,} filas (se esperaban {n:,})"
                                if got is not None else f"{table}: no generada")
    if problems:
        raise PipelineError("Validación fallida: " + "; ".join(problems))


# --------------------------------------------------------------------------- #
# Entrada desde/hacia el data lake
# --------------------------------------------------------------------------- #
RAW_PREFIXES = ("raw/roundrobin", "raw/evolutionary")  # NO raw_cloud/: duplicaría los datos


def run_from_lake(fs_client, spark: SparkSession, workdir: str | Path, io=None,
                  expected: dict[str, int] | None = EXPECTED_COUNTS) -> dict:
    """Descarga raw/ del lake, ejecuta los pipelines y sube curated/ y analytics/.

    Si la validación falla no se sube nada.
    """
    workdir = Path(workdir)
    raw_dir, out_dir = workdir / "raw", workdir / "out"
    for prefix in RAW_PREFIXES:
        files = download_prefix(fs_client, prefix, raw_dir / prefix.split("/", 1)[1])
        size = sum(f.stat().st_size for f in files)
        print(f"Descargados {len(files)} ficheros de {prefix}/ ({size / 1e6:.2f} MB)", flush=True)
        if not any(f.suffix == ".parquet" for f in files):
            raise PipelineError(f"No hay ficheros Parquet en {fs_client.file_system_name}/{prefix}")

    result = run_pipelines(spark, raw_dir, out_dir, io=io, expected=expected)

    n_files = 0
    for table, layer in TABLES.items():
        local = table_path(out_dir, table)
        files = sorted(local.rglob("*.parquet"))  # sin .crc ni _SUCCESS
        if not files:
            raise PipelineError(f"{table} no tiene ficheros Parquet en {local}")
        # Los part-*.parquet de Spark tienen nombres aleatorios: se borra la tabla
        # anterior para no mezclar ficheros de dos ejecuciones.
        delete_directory(fs_client, f"{layer}/{table}")
        upload_files(files, out_dir, "", fs_client)
        n_files += len(files)
    result["uploaded_files"] = n_files
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tfm_axelrod.spark_pipelines",
                                description="Pipelines Spark raw -> curated -> analytics.")
    p.add_argument("--raw", default="data/raw", help="Capa raw local (con roundrobin/ y evolutionary/)")
    p.add_argument("--out", default="data/spark", help="Carpeta de salida local (curated/ y analytics/)")
    p.add_argument("--from-lake", action="store_true",
                   help="Lee raw/ del lake y sube curated/ y analytics/ (ignora --raw y --out)")
    p.add_argument("--filesystem", default="lake", help="Sistema de ficheros ADLS Gen2 (por defecto: lake)")
    p.add_argument("--workdir", default=None, help="Carpeta temporal para --from-lake")
    p.add_argument("--io", choices=["auto", "spark", "arrow"], default="auto",
                   help="E/S de ficheros: spark (nativa) o arrow (Windows sin winutils)")
    p.add_argument("--master", default=None, help="Master de Spark (por defecto local[2])")
    p.add_argument("--driver-memory", default=None, help="Memoria de Spark (por defecto 5g)")
    p.add_argument("--no-check-counts", action="store_true",
                   help="No exige los recuentos del conjunto completo (para muestras)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    expected = None if args.no_check_counts else EXPECTED_COUNTS
    t0 = time.time()

    fs_client = None
    if args.from_lake:
        try:
            fs_client = get_filesystem_client(args.filesystem)
            check_filesystem(fs_client)
        except UploadError as exc:
            raise SystemExit(f"ERROR: {exc}") from exc

    spark = build_spark(args.master, args.driver_memory, io=args.io)
    io = make_io(args.io, spark)
    print(f"Spark {spark.version} master={spark.sparkContext.master} "
          f"memoria={spark.conf.get('spark.driver.memory')} E/S={io.name}", flush=True)
    try:
        if fs_client is not None:
            tmp = args.workdir or tempfile.mkdtemp(prefix="tfm-spark-")
            try:
                result = run_from_lake(fs_client, spark, tmp, io=io, expected=expected)
            finally:
                if not args.workdir:
                    shutil.rmtree(tmp, ignore_errors=True)
        else:
            result = run_pipelines(spark, args.raw, args.out, io=io, expected=expected)
    except (PipelineError, UploadError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    finally:
        spark.stop()

    counts = " ".join(f"{k}={v}" for k, v in result["counts"].items())
    uploaded = f" ficheros_subidos={result['uploaded_files']}" if "uploaded_files" in result else ""
    print(f"RESUMEN validacion=OK {counts}{uploaded} duracion={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
