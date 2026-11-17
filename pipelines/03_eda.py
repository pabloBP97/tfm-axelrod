# Databricks notebook source
# MAGIC %pip install adjustText -q
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC Celda 1: Configuración y carga

# COMMAND ----------

import os, shutil
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pyspark.sql import functions as F

CATALOG, SCHEMA = "workspace", "tfm_axelrod"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"
FIG_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/lake/analytics/figures"
os.makedirs(FIG_DIR, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

def save(fig, name):
    tmp = f"/tmp/{name}.png"
    fig.savefig(tmp, dpi=200, bbox_inches="tight")
    shutil.copy(tmp, f"{FIG_DIR}/{name}.png")
    plt.show()

rank = spark.table(T("analytics_strategy_ranking")).toPandas()
r200 = rank[rank.turns == 200]
base = r200[r200.noise == 0].sort_values("avg_score_per_turn", ascending=False)
order = base.strategy.tolist()

evo = spark.table(T("curated_evolution")).toPandas()
final = evo[evo.generation == evo.generation.max()]
evo_top = final.groupby("strategy").population_share.max().sort_values(ascending=False).head(4).index.tolist()

# Estrategias destacadas: mismo color en todas las figuras
candidates = ["Tit For Tat", "Defector", "Grudger"] + order[:3] + evo_top
HIGHLIGHT = list(dict.fromkeys(s for s in candidates if s in order))[:8]
COLOR = {s: SERIES[i] for i, s in enumerate(HIGHLIGHT)}
print("Destacadas:", HIGHLIGHT)

# COMMAND ----------

# MAGIC %md
# MAGIC Celda 2: Ranking y sensibilidad al ruido

# COMMAND ----------

# Figura 1: ranking sin ruido
fig, ax = plt.subplots(figsize=(8, 9))
d = base.iloc[::-1]
ax.barh(d.strategy, d.avg_score_per_turn, color=SERIES[0], height=0.7)
ax.set_xlabel("Puntuación media por ronda")
ax.set_title("Ranking de estrategias sin ruido (partidas de 200 rondas)")
ax.xaxis.grid(True); ax.set_axisbelow(True); ax.margins(y=0.01)
save(fig, "01_ranking_sin_ruido")

# Figura 2: puntuación por estrategia y nivel de ruido
pivot = r200.pivot(index="strategy", columns="noise", values="avg_score_per_turn").loc[order]
fig, ax = plt.subplots(figsize=(6.5, 10))
im = ax.imshow(pivot.values, cmap=SEQ, aspect="auto")
ax.set_xticks(range(pivot.shape[1]), [f"{c:g}" for c in pivot.columns])
ax.set_yticks(range(len(order)), order)
mid = (pivot.values.min() + pivot.values.max()) / 2
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        v = pivot.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="#ffffff" if v > mid else INK)
ax.set_xlabel("Nivel de ruido")
ax.set_title("Puntuación media por ronda según el ruido")
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
fig.colorbar(im, ax=ax, shrink=0.5, label="Puntuación media por ronda")
save(fig, "02_puntuacion_por_ruido")

# Figura 3: cambio de posición en el ranking con el ruido
rk = r200.pivot(index="strategy", columns="noise", values="rank")
x = list(range(len(rk.columns)))
fig, ax = plt.subplots(figsize=(8, 8))
for s in order:
    if s not in COLOR:
        ax.plot(x, rk.loc[s].values, color=AXIS, lw=1)
for s in HIGHLIGHT:
    ax.plot(x, rk.loc[s].values, color=COLOR[s], lw=2, marker="o", ms=6)
    ax.text(x[-1] + 0.08, rk.loc[s].values[-1], s, va="center", fontsize=9, color=INK2)
ax.set_xticks(x, [f"{c:g}" for c in rk.columns])
ax.set_xlim(-0.1, x[-1] + 1.0)
ax.set_yticks([1, 5, 10, 15, 20, 25, 30])
ax.invert_yaxis()
ax.set_xlabel("Nivel de ruido"); ax.set_ylabel("Posición en el ranking")
ax.set_title("Cómo cambia el ranking al aumentar el ruido")
ax.yaxis.grid(True); ax.set_axisbelow(True)
save(fig, "03_ranking_vs_ruido")

# COMMAND ----------

# MAGIC %md
# MAGIC Celda 3: Dinámica evolutiva y enfrentamientos directos

# COMMAND ----------

# Figura 4: dinámica evolutiva
noises = sorted(evo.noise.unique())
fig, axes = plt.subplots(1, len(noises), figsize=(14, 4.2), sharey=True)
for ax, nz in zip(axes, noises):
    d = evo[evo.noise == nz].pivot(index="generation", columns="strategy", values="population_share")
    for s in d.columns:
        if s not in COLOR:
            ax.plot(d.index, d[s], color=AXIS, lw=0.8)
    for s in HIGHLIGHT:
        if s in d.columns:
            ax.plot(d.index, d[s], color=COLOR[s], lw=2, label=s)
    ax.set_title(f"Ruido {nz:g}"); ax.set_xlabel("Generación")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
axes[0].set_ylabel("Cuota de población")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.16))
fig.suptitle("Dinámica evolutiva por nivel de ruido", x=0.01, y=1.06, ha="left", fontweight="bold")
save(fig, "04_dinamica_evolutiva")

# Figura 5: enfrentamientos directos sin ruido
h2h = (spark.table(T("analytics_match_summary")).filter("noise = 0 AND turns = 200")
       .groupBy("player", "opponent").agg(F.avg("score_per_turn").alias("s")).toPandas())
m = h2h.pivot(index="player", columns="opponent", values="s").loc[order, order]
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(m.values, cmap=SEQ)
ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
ax.set_yticks(range(len(order)), order, fontsize=7)
ax.set_xlabel("Rival"); ax.set_ylabel("Estrategia")
ax.set_title("Puntuación media por ronda de cada estrategia contra cada rival (sin ruido)")
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
fig.colorbar(im, ax=ax, shrink=0.6, label="Puntuación media por ronda")
save(fig, "05_enfrentamientos")

# COMMAND ----------

# MAGIC %md
# MAGIC Celda 4: Clústeres de comportamiento

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


ft = spark.table(T("analytics_classifier_features"))
META = ["match_key", "player_index", "player", "opponent", "noise", "turns", "repetition", "is_self_play", "n"]
feat_cols = [c for c in ft.columns if c not in META]

prof = (ft.filter("n = 100 AND noise = 0 AND NOT is_self_play")
        .groupBy("player").agg(*[F.avg(c).alias(c) for c in feat_cols])
        .toPandas().set_index("player").astype(float))

X = StandardScaler().fit_transform(prof.values)
sil = {k: silhouette_score(X, KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X))
       for k in range(2, 7)}
best_k = max(sil, key=sil.get)
prof["cluster"] = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(X)
pca = PCA(n_components=2, random_state=42)
xy = pca.fit_transform(X)

fig, ax = plt.subplots(figsize=(12, 9))
for c in sorted(prof.cluster.unique()):
    mask = (prof.cluster == c).values
    ax.scatter(xy[mask, 0], xy[mask, 1], s=70, color=SERIES[c], edgecolor=SURFACE,
               linewidth=1.5, label=f"Clúster {c + 1}", zorder=3)
from adjustText import adjust_text
texts = [ax.text(px, py, name, fontsize=7, color=INK2) for (px, py), name in zip(xy, prof.index)]
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6))

ax.set_xlabel(f"Componente principal 1 ({pca.explained_variance_ratio_[0]:.0%} de la varianza)")
ax.set_ylabel(f"Componente principal 2 ({pca.explained_variance_ratio_[1]:.0%} de la varianza)")
ax.set_title("Clústeres de comportamiento de las estrategias")
ax.grid(True); ax.set_axisbelow(True)
ax.legend(frameon=False)
save(fig, "06_clusteres")

# COMMAND ----------

# MAGIC %md
# MAGIC Celda 5: Resumen en texto

# COMMAND ----------

print("Silueta por k:", {k: round(v, 3) for k, v in sil.items()}, "-> k elegido:", best_k)
print("\nCLÚSTERES")
for c in sorted(prof.cluster.unique()):
    g = prof[prof.cluster == c]
    print(f"  Clúster {c + 1} (cooperación media {g.coop_rate.mean():.2f}): {', '.join(g.index)}")

print("\nTOP 5 POR NIVEL DE RUIDO (200 rondas)")
for nz, g in r200.groupby("noise"):
    top = g.sort_values("rank").head(5)
    print(f"  Ruido {nz:g}: " + "; ".join(f"{s} ({v:.2f})" for s, v in zip(top.strategy, top.avg_score_per_turn)))

print("\nÚLTIMOS 3 POR NIVEL DE RUIDO")
for nz, g in r200.groupby("noise"):
    print(f"  Ruido {nz:g}: " + ", ".join(g.sort_values("rank").tail(3).strategy))

print("\nCUOTA FINAL EN EL TORNEO EVOLUTIVO (top 5)")
for nz, g in final.groupby("noise"):
    top = g.sort_values("population_share", ascending=False).head(5)
    print(f"  Ruido {nz:g}: " + "; ".join(f"{s} ({v:.1%})" for s, v in zip(top.strategy, top.population_share)))