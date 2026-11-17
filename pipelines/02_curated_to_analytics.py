# Databricks notebook source
# MAGIC %md
# MAGIC Celda 1: Resumen por partida y ranking

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG, SCHEMA = "workspace", "tfm_axelrod"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"

moves = spark.table(T("curated_moves"))

match_summary = (
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
match_summary.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("analytics_match_summary"))

ranking = (
    spark.table(T("analytics_match_summary"))
    .filter(~F.col("is_self_play"))
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
ranking.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("analytics_strategy_ranking"))

print("Filas en analytics_match_summary:", spark.table(T("analytics_match_summary")).count())
print("Filas en analytics_strategy_ranking:", spark.table(T("analytics_strategy_ranking")).count())
(spark.table(T("analytics_strategy_ranking"))
    .filter("noise = 0 AND turns = 200 AND rank <= 5").orderBy("rank").show(truncate=False))

# COMMAND ----------

# MAGIC %md
# MAGIC Celda 2: Caracteristicas del clasificador

# COMMAND ----------

N_VALUES = [5, 10, 20, 30, 50, 75, 100]
STATES = {"cc": (1, 1), "cd": (1, 0), "dc": (0, 1), "dd": (0, 0)}  # (jugada propia, jugada del rival)
KEYS = ["match_key", "player_index", "player", "opponent", "noise", "turns", "repetition", "is_self_play"]

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

features = (
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
features.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("analytics_classifier_features"))

# COMMAND ----------

# MAGIC %md
# MAGIC Celda 3: Comprobación

# COMMAND ----------

ft = spark.table(T("analytics_classifier_features"))
print("Filas en analytics_classifier_features:", ft.count())
(ft.filter("player = 'Tit For Tat' AND opponent = 'Defector' AND noise = 0 "
           "AND turns = 100 AND repetition = 0 AND n = 10")
   .select("n", "coop_rate", "first_move_coop", "first_defection", "max_d_streak", "p_coop_after_dd")
   .show())