# Databricks notebook source
# MAGIC %md
# MAGIC Celda 1: Lectura y reglas de validación

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG, SCHEMA = "workspace", "tfm_axelrod"
RAW = f"/Volumes/{CATALOG}/{SCHEMA}/lake/raw"

raw = spark.read.parquet(f"{RAW}/roundrobin")

# Clave única de partida
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

checked = (
    raw.withColumn("match_key", match_key)
    .withColumn("ok_actions", F.coalesce(pa.isin("C", "D") & oa.isin("C", "D"), F.lit(False)))
    .withColumn("ok_round", F.coalesce(F.col("round").between(1, F.col("turns")), F.lit(False)))
    .withColumn("ok_payoff", F.coalesce(F.col("player_payoff") == expected_payoff, F.lit(False)))
    .withColumn("rows_in_round", F.count("*").over(Window.partitionBy("match_key", "round")))
    .withColumn("rows_in_match", F.count("*").over(Window.partitionBy("match_key")))
    .withColumn("ok_shape", (F.col("rows_in_round") == 2) & (F.col("rows_in_match") == 2 * F.col("turns")))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Celda 2: Informe de calidad y escritura de la tabla

# COMMAND ----------

report = checked.agg(
    F.count("*").alias("filas_raw"),
    F.sum((~F.col("ok_actions")).cast("int")).alias("acciones_invalidas"),
    F.sum((~F.col("ok_round")).cast("int")).alias("rondas_fuera_de_rango"),
    F.sum((~F.col("ok_payoff")).cast("int")).alias("pagos_incoherentes"),
    F.sum((~F.col("ok_shape")).cast("int")).alias("partidas_incompletas"),
)
report.show()

curated = (
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

(curated.write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.curated_moves"))

print("Filas en curated_moves:", spark.table(f"{CATALOG}.{SCHEMA}.curated_moves").count())

# COMMAND ----------

# MAGIC %md
# MAGIC Celda 3: Torneos evolutivos

# COMMAND ----------

evo = spark.read.parquet(f"{RAW}/evolutionary")

sums = evo.groupBy("tournament_id", "noise", "generation").agg(F.sum("population_share").alias("total"))
bad = sums.filter(~F.col("total").between(0.999, 1.001)).count()
print("Generaciones cuya población no suma 1:", bad)

(evo.withColumn("curated_at", F.current_timestamp())
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.curated_evolution"))

print("Filas en curated_evolution:", spark.table(f"{CATALOG}.{SCHEMA}.curated_evolution").count())
evo.groupBy("noise").agg(F.countDistinct("generation").alias("generaciones")).orderBy("noise").show()